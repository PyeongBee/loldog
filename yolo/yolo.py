# 1. 라이브러리 설치
# pip install ultralytics onnx

from ultralytics import YOLO

# 2. 모델 로드 (yolov8n 또는 yolo11n 선택)
model = YOLO("./runs/detect/train/weights/best.pt")

# 3. ONNX로 내보내기
# imgsz=[320, 320] -> 입력 크기 지정
# opset=12 -> IR version 11 이상을 보장하는 현대적인 설정
path = model.export(format="onnx", imgsz=[1858, 1858], opset=12)

print(f"모델 저장 완료: {path}")
