import { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
  LayoutRectangle,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useLocalSearchParams, useRouter } from "expo-router";
import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import { Asset } from "expo-asset";
import jpeg from "jpeg-js";
import { Buffer } from "buffer";

type OrtModule = typeof import("onnxruntime-react-native");
type OrtSession = Awaited<ReturnType<OrtModule["InferenceSession"]["create"]>>;
type OrtTensor = InstanceType<OrtModule["Tensor"]>;

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

const MODEL_ASSET = require("../assets/model/yolo11n.onnx");
const INPUT_SIZE = 320; // YOLO 입력 크기 (가벼운 모델 권장)
const CONF_THRESHOLD = 0.35;
const NMS_THRESHOLD = 0.45;

export default function DetectScreen() {
  const router = useRouter();
  const { champion } = useLocalSearchParams<{ champion?: string }>();
  const [permission, requestPermission] = useCameraPermissions();
  const [flash, setFlash] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [previewLayout, setPreviewLayout] = useState<LayoutRectangle | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelInitTried, setModelInitTried] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [ort, setOrt] = useState<OrtModule | null>(null);
  const [session, setSession] = useState<OrtSession | null>(null);
  const [status, setStatus] = useState("모델 준비 중...");
  const cameraRef = useRef<CameraView>(null);
  const flashTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const championLabel = useMemo(
    () => (typeof champion === "string" ? champion : "선택되지 않음"),
    [champion]
  );

  useEffect(() => {
    return () => {
      if (flashTimeout.current) {
        clearTimeout(flashTimeout.current);
      }
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      setDetections((prev) => prev.filter((d) => d.expiresAt > now));
    }, 400);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const prepareModel = async () => {
      if (modelInitTried || modelReady || isModelLoading) return;
      setModelInitTried(true);
      setIsModelLoading(true);
      setStatus("모델 로딩 중...");
      try {
        // NOTE: onnxruntime-react-native는 import 시점에 네이티브(JNI/ObjC) 설치를 시도합니다.
        // Expo Go 등 네이티브 모듈이 없는 환경에서는 여기서 실패하므로 동적 import로 감싸서 크래시를 방지합니다.
        const ortModule = ort ?? (await import("onnxruntime-react-native"));
        if (!ort) setOrt(ortModule);

        const asset = Asset.fromModule(MODEL_ASSET);
        await asset.downloadAsync(); // localUri 확보
        const modelPath = asset.localUri ?? asset.uri;
        const loadedSession = await ortModule.InferenceSession.create(modelPath, {
          executionProviders: ["cpu"],
        });
        setSession(loadedSession);
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
  }, [isModelLoading, modelReady, ort]);

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
    if (!cameraRef.current || !session || !ort) {
      setStatus(
        !ort ? "onnxruntime 로딩 실패: Dev Client가 필요합니다." : "세션이 준비되지 않았습니다."
      );
      return;
    }
    setIsRunning(true);
    setStatus("1회 감지 중...");

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.4,
        base64: true,
        skipProcessing: true,
      });
      if (!photo.base64) {
        setStatus("캡처 실패");
        return;
      }

      // 입력 크기로 리사이즈 후 base64 획득
      const resized = await manipulateAsync(
        photo.uri,
        [{ resize: { width: INPUT_SIZE, height: INPUT_SIZE } }],
        { base64: true, compress: 0.6, format: SaveFormat.JPEG }
      );
      if (!resized.base64) {
        setStatus("리사이즈 실패");
        return;
      }

      const tensor = await buildInputTensor(ort, resized.base64, INPUT_SIZE);
      const feeds: Record<string, OrtTensor> = {};
      const inputName = session.inputNames[0];
      feeds[inputName] = tensor;

      const results = await session.run(feeds);
      const outputName = session.outputNames[0];
      const output = results[outputName];
      const dets = parseYoloOutput(output.data as Float32Array, output.dims, {
        inputSize: INPUT_SIZE,
        confThreshold: CONF_THRESHOLD,
      });
      const finalDets = nonMaxSuppression(dets, NMS_THRESHOLD);

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

      <View style={styles.actions}>
        <Pressable
          style={[
            styles.primaryButton,
            (!modelReady || isModelLoading || isRunning) && styles.primaryButtonDisabled,
          ]}
          onPress={runYoloOnce}
          disabled={!modelReady || isModelLoading || isRunning}
        >
          <Text style={styles.primaryLabel}>
            {isRunning ? "감지 중..." : modelReady ? "YOLO 1회 감지" : "모델 준비 중"}
          </Text>
        </Pressable>
        <Pressable style={styles.secondaryButton} onPress={triggerMockDetection}>
          <Text style={styles.secondaryLabel}>모의 감지(랜덤)</Text>
        </Pressable>
        <Pressable style={styles.secondaryButton} onPress={() => router.back()}>
          <Text style={styles.secondaryLabel}>다시 선택</Text>
        </Pressable>
      </View>
    </View>
  );
}

async function buildInputTensor(ort: OrtModule, base64: string, size: number): Promise<OrtTensor> {
  const buffer = Buffer.from(base64, "base64");
  const decoded = jpeg.decode(buffer, { useTArray: true });
  if (decoded.width !== size || decoded.height !== size) {
    throw new Error(`입력 크기 불일치: ${decoded.width}x${decoded.height}`);
  }
  const { data } = decoded; // RGBA
  const floats = new Float32Array(size * size * 3);
  let di = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    floats[di++] = r;
    floats[di++] = g;
    floats[di++] = b;
  }
  // CHW
  const chw = new Float32Array(size * size * 3);
  const channelSize = size * size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (y * size + x) * 3;
      const c = y * size + x;
      chw[c] = floats[idx];
      chw[c + channelSize] = floats[idx + 1];
      chw[c + channelSize * 2] = floats[idx + 2];
    }
  }
  return new ort.Tensor("float32", chw, [1, 3, size, size]);
}

function parseYoloOutput(
  data: Float32Array,
  dims: readonly number[] | undefined,
  opts: { inputSize: number; confThreshold: number }
) {
  const detections: Omit<Detection, "id" | "expiresAt">[] = [];
  if (!dims || dims.length < 3) {
    return detections;
  }
  const [batch, anchors, features] = dims;
  if (batch !== 1) return detections;
  const stride = features ?? 85;
  for (let i = 0; i < anchors; i++) {
    const offset = i * stride;
    const obj = data[offset + 4];
    if (obj < opts.confThreshold) continue;
    let bestCls = 0;
    let bestScore = 0;
    for (let c = 5; c < stride; c++) {
      const clsScore = data[offset + c];
      if (clsScore > bestScore) {
        bestScore = clsScore;
        bestCls = c - 5;
      }
    }
    const score = obj * bestScore;
    if (score < opts.confThreshold) continue;
    const cx = data[offset];
    const cy = data[offset + 1];
    const w = data[offset + 2];
    const h = data[offset + 3];
    const x = (cx - w / 2) / opts.inputSize;
    const y = (cy - h / 2) / opts.inputSize;
    detections.push({
      x,
      y,
      w: w / opts.inputSize,
      h: h / opts.inputSize,
      score,
      label: `cls${bestCls}`,
    });
  }
  return detections;
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
