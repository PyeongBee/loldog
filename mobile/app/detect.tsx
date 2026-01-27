import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
  LayoutRectangle,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useRouter } from "expo-router";
import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import jpeg from "jpeg-js";
import { Buffer } from "buffer";
import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";

type Detection = {
  id: string;
  x: number; // 0-1 normalized (left)
  y: number; // 0-1 normalized (top)
  w: number; // 0-1 normalized width
  h: number; // 0-1 normalized height
  score: number;
  label: string;
  expiresAt: number;
};

const MODEL_ASSET = require("../assets/model/11n_best_epoch300_train.tflite");
const INPUT_SIZE = 640; // YOLO 입력 크기 (가벼운 모델 권장)
const CONF_THRESHOLD = 0.7;
const NMS_THRESHOLD = 0.3;
const MAX_DETECTIONS = 10;
const MIN_ACCEPT_SCORE = 0.2;
const AUTO_DETECT_INTERVAL_MS = 1000;

const LEESIN_LABELS = ["Leesin"];

export default function DetectScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [permission, requestPermission] = useCameraPermissions();
  const [flash, setFlash] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [previewLayout, setPreviewLayout] = useState<LayoutRectangle | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelInitTried, setModelInitTried] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [model, setModel] = useState<TensorflowModel | null>(null);
  const [status, setStatus] = useState("모델 준비 중...");
  const cameraRef = useRef<CameraView>(null);
  const flashTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const autoInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const isRunningRef = useRef(false);

  const championLabel = "Leesin";

  useEffect(() => {
    return () => {
      if (flashTimeout.current) {
        clearTimeout(flashTimeout.current);
      }
    };
  }, []);

  useEffect(() => {
    isRunningRef.current = isRunning;
  }, [isRunning]);

  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      setDetections((prev) => prev.filter((d) => d.expiresAt > now));
    }, 400);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (autoInterval.current) {
      clearInterval(autoInterval.current);
      autoInterval.current = null;
    }
    if (!permission?.granted || !modelReady || !model) {
      return;
    }
    autoInterval.current = setInterval(() => {
      if (isRunningRef.current) return;
      runYoloOnce();
    }, AUTO_DETECT_INTERVAL_MS);

    return () => {
      if (autoInterval.current) {
        clearInterval(autoInterval.current);
        autoInterval.current = null;
      }
    };
  }, [permission?.granted, modelReady, model]);

  useEffect(() => {
    const prepareModel = async () => {
      if (modelInitTried || modelReady || isModelLoading) return;
      setModelInitTried(true);
      setIsModelLoading(true);
      setStatus("TFLite 모델 로딩 중...");
      try {
        const loadedModel = await loadTensorflowModel(MODEL_ASSET);
        console.log("TFLite inputs", loadedModel.inputs);
        console.log("TFLite outputs", loadedModel.outputs);
        setModel(loadedModel);
        setModelReady(true);
        setStatus("모델 준비 완료");
      } catch (err) {
        console.warn("모델 준비 실패", err);
        setStatus("모델 준비 실패: Dev Client(네이티브 빌드)로 실행 중인지 확인하세요.");
      } finally {
        setIsModelLoading(false);
      }
    };

    prepareModel();
  }, [isModelLoading, modelReady, model]);

  const triggerMockDetection = () => {
    const now = Date.now();
    setDetections((prev) => [
      ...prev,
      {
        id: `${now}`,
        x: Math.random() * 0.6 + 0.2,
        y: Math.random() * 0.6 + 0.2,
        w: 0.12,
        h: 0.12,
        score: 0.9,
        label: "mock",
        expiresAt: now + 5000,
      },
    ]);
    setFlash(true);
    if (flashTimeout.current) clearTimeout(flashTimeout.current);
    flashTimeout.current = setTimeout(() => setFlash(false), 300);
  };

  const runYoloOnce = async () => {
    if (!cameraRef.current || !model) {
      setStatus("TFLite 모델이 준비되지 않았습니다.");
      return;
    }
    setIsRunning(true);
    setStatus("자동 감지 중...");

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.4,
        base64: true,
        skipProcessing: true,
        shutterSound: false,
      });
      if (!photo.base64) {
        setStatus("캡처 실패");
        return;
      }

      // 입력 크기로 리사이즈 후 base64 획득
      const inputSpec = getModelInputSpec(model, INPUT_SIZE);
      const resized = await manipulateAsync(
        photo.uri,
        [{ resize: { width: inputSpec.width, height: inputSpec.height } }],
        { base64: true, compress: 0.6, format: SaveFormat.JPEG }
      );
      if (!resized.base64) {
        setStatus("리사이즈 실패");
        return;
      }

      const inputBuffer = await buildInputBuffer(
        resized.base64,
        inputSpec.width,
        inputSpec.height,
        inputSpec.layout,
        inputSpec.dataType
      );
      if (!inputBuffer || inputBuffer.length === 0) {
        throw new Error("입력 버퍼가 비어 있습니다.");
      }
      const results = await model.run([inputBuffer]);
      const output = results[0];
      const outputDims = model.outputs?.[0]?.shape;
      if (!(output instanceof Float32Array)) {
        throw new Error(
          `출력 타입이 Float32Array가 아닙니다: ${output?.constructor?.name ?? "unknown"}`
        );
      }
      console.log("yolo output dims", outputDims);
      const maxScore = getYoloMaxScore(
        output,
        outputDims,
        LEESIN_LABELS.length
      );
      console.log("yolo max score", maxScore);
      if (maxScore < MIN_ACCEPT_SCORE) {
        setStatus("감지 없음");
        return;
      }
      const dets = parseYoloOutput(output, outputDims, {
        inputSize: inputSpec.width,
        confThreshold: CONF_THRESHOLD,
        labels: LEESIN_LABELS,
      });
      const finalDets = nonMaxSuppression(dets, NMS_THRESHOLD).slice(0, MAX_DETECTIONS);

      if (finalDets.length > 0) {
        const now = Date.now();
        setDetections((prev) => [
          ...prev,
          ...finalDets.map((d, idx) => ({
            id: `${now}-${idx}`,
            ...d,
            expiresAt: now + 5000,
          })),
        ]);
        setFlash(true);
        if (flashTimeout.current) clearTimeout(flashTimeout.current);
        flashTimeout.current = setTimeout(() => setFlash(false), 300);
        setStatus(`감지됨: ${finalDets.length}개`);
      } else {
        setStatus("감지 없음");
      }
    } catch (err) {
      console.warn("YOLO 추론 실패", err);
      setStatus("추론 실패: 모델/입력 확인 필요");
    } finally {
      setIsRunning(false);
    }
  };

  if (!permission) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.centered}>
        <Text style={styles.permissionText}>카메라 권한이 필요합니다.</Text>
        <Pressable style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.primaryLabel}>권한 요청</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>{championLabel} 감지 중</Text>
        <Text style={styles.caption}>미니맵이 화면에 잘 보이도록 맞춰주세요.</Text>
        <Text style={styles.caption}>{status}</Text>
      </View>

      <View style={styles.preview} onLayout={(event) => setPreviewLayout(event.nativeEvent.layout)}>
        <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
        {flash && <View style={styles.flashOverlay} pointerEvents="none" />}
        {previewLayout &&
          detections.map((detection) => (
            <View
              key={detection.id}
              style={[
                styles.box,
                {
                  left: detection.x * previewLayout.width,
                  top: detection.y * previewLayout.height,
                  width: detection.w * previewLayout.width,
                  height: detection.h * previewLayout.height,
                },
              ]}
              pointerEvents="none"
            >
              <Text style={styles.boxLabel}>
                {detection.label} {detection.score.toFixed(2)}
              </Text>
            </View>
          ))}
      </View>

      <View style={[styles.actions, { paddingBottom: 16 + insets.bottom }]}>
        <View
          style={[
            styles.primaryButton,
            (!modelReady || isModelLoading) && styles.primaryButtonDisabled,
          ]}
        >
          <Text style={styles.primaryLabel}>
            {!modelReady || isModelLoading
              ? "모델 준비 중"
              : isRunning
                ? "자동 감지 중..."
                : "자동 감지 대기"}
          </Text>
        </View>
        <Pressable style={styles.secondaryButton} onPress={() => router.back()}>
          <Text style={styles.secondaryLabel}>다시 선택</Text>
        </Pressable>
      </View>
    </View>
  );
}

type InputLayout = "NHWC" | "CHW";
type InputDataType = TensorflowModel["inputs"][number]["dataType"];
type InputSpec = {
  width: number;
  height: number;
  layout: InputLayout;
  dataType: InputDataType | undefined;
};

function getModelInputSpec(model: TensorflowModel, fallbackSize: number): InputSpec {
  const shape = model.inputs?.[0]?.shape ?? [];
  const isCHW = shape.length === 4 && shape[1] === 3;
  const isNHWC = shape.length === 4 && shape[3] === 3;
  const width = isNHWC ? shape[2] : isCHW ? shape[3] : fallbackSize;
  const height = isNHWC ? shape[1] : isCHW ? shape[2] : fallbackSize;
  const layout: InputLayout = isNHWC ? "NHWC" : "CHW";
  const dataType = model.inputs?.[0]?.dataType;
  return {
    width: Number.isFinite(width) && width > 0 ? width : fallbackSize,
    height: Number.isFinite(height) && height > 0 ? height : fallbackSize,
    layout,
    dataType,
  };
}

async function buildInputBuffer(
  base64: string,
  width: number,
  height: number,
  layout: InputLayout,
  dataType: InputDataType | undefined
): Promise<Float32Array | Uint8Array> {
  const buffer = Buffer.from(base64, "base64");
  const decoded = jpeg.decode(buffer, { useTArray: true });
  if (decoded.width !== width || decoded.height !== height) {
    throw new Error(`입력 크기 불일치: ${decoded.width}x${decoded.height}`);
  }
  const { data } = decoded; // RGBA
  const useFloat32 = dataType === "float32" || dataType === undefined;
  const useUint8 = dataType === "uint8";
  if (!useFloat32 && !useUint8) {
    throw new Error(`지원하지 않는 입력 타입: ${dataType ?? "unknown"}`);
  }

  const floats = useFloat32 ? new Float32Array(width * height * 3) : null;
  const bytes = useUint8 ? new Uint8Array(width * height * 3) : null;
  let di = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    if (useFloat32 && floats) {
      floats[di++] = r / 255;
      floats[di++] = g / 255;
      floats[di++] = b / 255;
    } else if (useUint8 && bytes) {
      bytes[di++] = r;
      bytes[di++] = g;
      bytes[di++] = b;
    }
  }

  if (layout === "NHWC") {
    return (useFloat32 ? floats : bytes) as Float32Array | Uint8Array;
  }

  // CHW
  const chw = useFloat32
    ? new Float32Array(width * height * 3)
    : new Uint8Array(width * height * 3);
  const channelSize = width * height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 3;
      const c = y * width + x;
      if (useFloat32 && floats) {
        (chw as Float32Array)[c] = floats[idx];
        (chw as Float32Array)[c + channelSize] = floats[idx + 1];
        (chw as Float32Array)[c + channelSize * 2] = floats[idx + 2];
      } else if (useUint8 && bytes) {
        (chw as Uint8Array)[c] = bytes[idx];
        (chw as Uint8Array)[c + channelSize] = bytes[idx + 1];
        (chw as Uint8Array)[c + channelSize * 2] = bytes[idx + 2];
      }
    }
  }
  return chw;
}

function parseYoloOutput(
  data: Float32Array,
  dims: readonly number[] | undefined,
  opts: { inputSize: number; confThreshold: number; labels?: string[] }
) {
  const detections: Omit<Detection, "id" | "expiresAt">[] = [];
  if (!dims || dims.length < 3) {
    return detections;
  }
  const [batch, dim1, dim2] = dims;
  if (batch !== 1) return detections;
  const featuresFirst = dim1 <= 100 && dim2 > 1000;
  const anchors = featuresFirst ? dim2 : dim1;
  const features = featuresFirst ? dim1 : dim2;
  const labelsCount = opts.labels?.length ?? null;
  const hasObjectness =
    labelsCount !== null
      ? features - 5 === labelsCount
        ? true
        : features - 4 === labelsCount
          ? false
          : features >= 5
      : features >= 5;
  const classStart = hasObjectness ? 5 : 4;
  const numClasses = features - classStart;
  if (numClasses <= 0) return detections;

  const getValue = (anchorIdx: number, featureIdx: number) =>
    featuresFirst ? data[featureIdx * anchors + anchorIdx] : data[anchorIdx * features + featureIdx];

  let coordScale = opts.inputSize;
  const sampleCount = Math.min(anchors, 200);
  let maxCoord = 0;
  for (let i = 0; i < sampleCount; i++) {
    const cx = getValue(i, 0);
    const cy = getValue(i, 1);
    const w = getValue(i, 2);
    const h = getValue(i, 3);
    if (cx > maxCoord) maxCoord = cx;
    if (cy > maxCoord) maxCoord = cy;
    if (w > maxCoord) maxCoord = w;
    if (h > maxCoord) maxCoord = h;
  }
  if (maxCoord <= 1.5) {
    coordScale = 1;
  }

  for (let i = 0; i < anchors; i++) {
    const obj = hasObjectness ? getValue(i, 4) : 1;
    if (obj < opts.confThreshold) continue;
    let bestCls = 0;
    let bestScore = 0;
    for (let c = 0; c < numClasses; c++) {
      const clsScore = getValue(i, classStart + c);
      if (clsScore > bestScore) {
        bestScore = clsScore;
        bestCls = c;
      }
    }
    const score = obj * bestScore;
    if (score < opts.confThreshold) continue;
    const cx = getValue(i, 0);
    const cy = getValue(i, 1);
    const w = getValue(i, 2);
    const h = getValue(i, 3);
    if (!Number.isFinite(cx + cy + w + h)) continue;
    const x = (cx - w / 2) / coordScale;
    const y = (cy - h / 2) / coordScale;
    detections.push({
      x,
      y,
      w: w / coordScale,
      h: h / coordScale,
      score,
      label: opts.labels?.[bestCls] ?? `cls${bestCls}`,
    });
  }
  return detections;
}

function getYoloMaxScore(
  data: Float32Array,
  dims: readonly number[] | undefined,
  labelsCount: number
) {
  if (!dims || dims.length < 3) return 0;
  const [batch, dim1, dim2] = dims;
  if (batch !== 1) return 0;
  const featuresFirst = dim1 <= 100 && dim2 > 1000;
  const anchors = featuresFirst ? dim2 : dim1;
  const features = featuresFirst ? dim1 : dim2;
  const hasObjectness =
    features - 5 === labelsCount
      ? true
      : features - 4 === labelsCount
        ? false
        : features >= 5;
  const classStart = hasObjectness ? 5 : 4;
  const numClasses = features - classStart;
  if (numClasses <= 0) return 0;

  const getValue = (anchorIdx: number, featureIdx: number) =>
    featuresFirst ? data[featureIdx * anchors + anchorIdx] : data[anchorIdx * features + featureIdx];

  let maxScore = 0;
  for (let i = 0; i < anchors; i++) {
    const obj = hasObjectness ? getValue(i, 4) : 1;
    let bestScore = 0;
    for (let c = 0; c < numClasses; c++) {
      const clsScore = getValue(i, classStart + c);
      if (clsScore > bestScore) bestScore = clsScore;
    }
    const score = obj * bestScore;
    if (score > maxScore) maxScore = score;
  }
  return maxScore;
}

function nonMaxSuppression(dets: Omit<Detection, "id" | "expiresAt">[], iouThresh: number) {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const picked: typeof sorted = [];
  while (sorted.length) {
    const current = sorted.shift()!;
    picked.push(current);
    const rest = [];
    for (const det of sorted) {
      if (iou(current, det) < iouThresh) {
        rest.push(det);
      }
    }
    sorted.splice(0, sorted.length, ...rest);
  }
  return picked;
}

function iou(a: Omit<Detection, "id" | "expiresAt">, b: Omit<Detection, "id" | "expiresAt">) {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const interX1 = Math.max(a.x, b.x);
  const interY1 = Math.max(a.y, b.y);
  const interX2 = Math.min(ax2, bx2);
  const interY2 = Math.min(ay2, by2);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interArea = interW * interH;
  const union = a.w * a.h + b.w * b.h - interArea;
  return union <= 0 ? 0 : interArea / union;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0b1115",
  },
  header: {
    paddingHorizontal: 20,
    paddingTop: 18,
    paddingBottom: 12,
  },
  title: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "700",
    marginBottom: 4,
  },
  caption: {
    color: "#c6d0d9",
    fontSize: 14,
  },
  preview: {
    flex: 1,
    marginHorizontal: 16,
    marginTop: 8,
    borderRadius: 18,
    overflow: "hidden",
    backgroundColor: "#111a22",
    borderWidth: 1,
    borderColor: "#233144",
  },
  flashOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(255, 0, 0, 0.2)",
  },
  box: {
    position: "absolute",
    borderWidth: 2,
    borderColor: "#ff4d4f",
    backgroundColor: "rgba(255, 77, 79, 0.08)",
  },
  boxLabel: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "700",
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingHorizontal: 4,
    paddingVertical: 2,
  },
  actions: {
    padding: 16,
    gap: 10,
  },
  primaryButton: {
    backgroundColor: "#ff4d4f",
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
  },
  primaryButtonDisabled: {
    backgroundColor: "#3a4756",
  },
  primaryLabel: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "700",
  },
  secondaryButton: {
    backgroundColor: "#1b2732",
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#253343",
  },
  secondaryLabel: {
    color: "#d4e0eb",
    fontSize: 15,
    fontWeight: "600",
  },
  centered: {
    flex: 1,
    backgroundColor: "#0b1115",
    alignItems: "center",
    justifyContent: "center",
    gap: 16,
    padding: 24,
  },
  permissionText: {
    color: "#c6d0d9",
    fontSize: 15,
    textAlign: "center",
  },
});
