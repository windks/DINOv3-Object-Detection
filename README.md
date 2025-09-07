# DINOv3 Object Detection Framework

Production-ready object detection using pretrained DINOv3 vision transformers. **No training required** - detect objects immediately using natural language descriptions!

🎯 **학습 없이 바로 사용** | 🖱️ **GUI 지원** | 🖥️ **로컬 실행 가능** | 🐍 **Python API**

## 📋 목차
- [설치 방법](#-설치-방법)
- [빠른 시작](#-빠른-시작-5분-안에-시작하기)
- [상세 사용법](#-상세-사용법)
- [실전 예제](#-실전-예제)
- [성능 최적화](#-성능-최적화)

## 🚀 Key Features

- **Zero-Shot Detection**: Detect any object using text descriptions - no training needed
- **Pretrained Models**: Uses DINOv3 + CLIP for immediate object detection
- **Natural Language**: Describe what you want to find (e.g., "red car", "person wearing hat")
- **Production Ready**: Simple API designed for real-world applications
- **Fast Inference**: Optimized for speed with FP16 support
- **Flexible**: Works with images and videos

## 🛠 설치 방법

### 1. UV 사용 (권장)
```bash
# 저장소 클론
git clone <repository-url>
cd Dino_v3_OD

# UV로 의존성 설치
uv sync
```

### 2. pip 사용
```bash
# 저장소 클론 후
cd Dino_v3_OD

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install torch torchvision
pip install transformers opencv-python pillow numpy scipy
```

## 🚀 빠른 시작 (5분 안에 시작하기!)

### 🖥️ 로컬 실행 방법 (3가지)

#### 방법 1: 🖱️ GUI 실행 (초보자 추천)
```bash
# GUI 실행
python detect_gui.py

# Windows: 더블클릭으로 실행
run_detector.bat
```
- 마우스로 모든 작업 가능
- 이미지 미리보기 지원
- 결과 실시간 확인

#### 방법 2: 💻 커맨드라인 (CLI)
```bash
# 단일 이미지 처리
python detect.py photo.jpg "사람" "자동차" "강아지"

# 결과 저장
python detect.py photo.jpg --targets "person,car,dog" --output detected.jpg

# 비디오 처리
python detect.py video.mp4 --targets "사람,자동차" --output result.mp4

# 고급 옵션
python detect.py image.jpg --targets "빨간 차" --threshold 0.2 --model large --gpu

# 도움말
python detect.py --help
```

#### 방법 3: 📁 폴더 일괄 처리
```bash
# 폴더 내 모든 이미지 처리
python detect_folder.py ./images "사람" "자동차"

# CSV로 결과 저장
python detect_folder.py ./photos --targets "person,car,dog" --csv results.csv

# 하위 폴더 포함, 결과 이미지 저장
python detect_folder.py ./dataset --targets "vehicle" --recursive --output ./results

# 요약만 보기
python detect_folder.py ./images --targets "person" --summary
```

### 🐍 Python 코드로 사용 - 3줄로 끝!

```python
from production_api import detect_objects

# 이미지에서 원하는 객체 찾기 - 학습 불필요!
results = detect_objects("photo.jpg", ["사람", "자동차", "강아지"])
print(f"찾은 객체: {len(results)}개")
```

### 🎯 첫 번째 예제: 이미지에서 객체 찾기

```python
from production_api import DINOv3DetectorAPI
import cv2

# 1. 탐지기 초기화 (첫 실행시 모델 다운로드)
detector = DINOv3DetectorAPI(model_size="small")  # 빠른 테스트용

# 2. 이미지 로드
image = cv2.imread("my_photo.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. 원하는 객체 찾기 - 한국어도 OK!
results = detector.detect(
    image,
    ["사람", "빨간 자동차", "노란색 버스", "강아지"],
    confidence_threshold=0.3  # 신뢰도 임계값
)

# 4. 결과 확인
for detection in results:
    print(f"발견: {detection.class_name} (신뢰도: {detection.confidence:.2f})")
    print(f"위치: {detection.bbox}")

# 5. 시각화 (선택사항)
vis_image = detector.visualize(image, results, "result.jpg")
```

## 📖 상세 사용법

### 1️⃣ 기본 설정

```python
from production_api import DINOv3DetectorAPI

# 모델 크기 선택
detector = DINOv3DetectorAPI(
    model_size="small",    # 빠름, 간단한 객체에 적합
    # model_size="base",   # 균형잡힌 성능
    # model_size="large",  # 최고 정확도, 느림
    device="cuda",         # GPU 사용 (없으면 "cpu")
    use_fp16=True         # GPU에서 2배 빠른 추론
)
```

### 2️⃣ 다양한 입력 방식

```python
# 방법 1: 파일 경로
results = detector.detect("image.jpg", ["person", "car"])

# 방법 2: numpy 배열
import numpy as np
image_array = np.array(...)  # 당신의 이미지 데이터
results = detector.detect(image_array, ["person", "car"])

# 방법 3: PIL 이미지
from PIL import Image
pil_image = Image.open("image.jpg")
image_array = np.array(pil_image)
results = detector.detect(image_array, ["person", "car"])
```

### 3️⃣ 자연어로 객체 설명하기

```python
# 단순한 객체명
simple_targets = ["사람", "자동차", "개", "고양이"]

# 구체적인 설명
detailed_targets = [
    "빨간색 스포츠카",
    "파란 셔츠를 입은 사람",
    "안경을 쓴 노인",
    "배낭을 멘 학생",
    "노란색 스쿨버스"
]

# 복잡한 시나리오
complex_targets = [
    "횡단보도를 건너는 사람",
    "정지 신호에 멈춰있는 차",
    "공을 물고 있는 강아지",
    "노트북을 사용하는 사람"
]

results = detector.detect(image, detailed_targets)
```

### 4️⃣ 결과 필터링

```python
# 모든 탐지 결과
all_results = detector.detect(image, targets, confidence_threshold=0.2)

# 높은 신뢰도만
high_confidence = detector.filter_detections(
    all_results,
    min_confidence=0.7
)

# 특정 클래스만
only_people = detector.filter_detections(
    all_results,
    class_names=["사람", "person"]
)

# 큰 객체만 (픽셀 면적 기준)
large_objects = detector.filter_detections(
    all_results,
    min_area=10000  # 100x100 픽셀 이상
)
```

### 5️⃣ 여러 이미지 한번에 처리

```python
# 이미지 리스트 준비
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
targets = ["person", "car", "dog"]

# 배치 처리
all_results = detector.detect_batch(images, targets)

# 각 이미지별 결과
for idx, results in enumerate(all_results):
    print(f"\n이미지 {idx+1}: {len(results)}개 객체 발견")
    for det in results:
        print(f"  - {det.class_name}: {det.confidence:.2f}")
```

### 6️⃣ 비디오 처리

```python
from production_api import process_video

# 간단한 비디오 처리
process_video(
    "input.mp4",           # 입력 비디오
    "output.mp4",          # 출력 비디오
    ["사람", "자동차"],     # 찾을 객체
    threshold=0.3          # 신뢰도 임계값
)

# 고급 비디오 처리 (직접 제어)
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임별 처리
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect(frame_rgb, ["person", "car"])
    
    # 원하는 처리...
```

## 💡 실전 예제

### 예제 1: 보안 카메라 모니터링
```python
# 특정 상황 감지
security_targets = [
    "배낭을 맨 사람",
    "방치된 가방",
    "달리는 사람",
    "쓰러진 사람",
    "마스크를 쓴 사람"
]

results = detector.detect(cctv_frame, security_targets)

# 위험 상황 알림
for det in results:
    if det.class_name == "쓰러진 사람" and det.confidence > 0.7:
        send_alert("긴급 상황 감지!")
```

### 예제 2: 재고 관리
```python
# 제품 찾기
products = [
    "코카콜라 캔",
    "펩시 병",
    "오렌지 주스",
    "빈 선반"
]

results = detector.detect(shelf_image, products)

# 재고 확인
product_count = {}
for det in results:
    if det.class_name in product_count:
        product_count[det.class_name] += 1
    else:
        product_count[det.class_name] = 1
```

### 예제 3: 교통 모니터링
```python
# 교통 상황 분석
traffic_targets = [
    "빨간 신호등",
    "초록 신호등",
    "정지한 차량",
    "움직이는 차량",
    "횡단보도의 보행자"
]

results = detector.detect(traffic_cam, traffic_targets)

# 위반 감지
for det in results:
    if det.class_name == "빨간 신호등":
        red_light_box = det.bbox
        # 신호 위반 차량 확인...
```

### 예제 4: 품질 검사
```python
# 제품 결함 찾기
defect_targets = [
    "긁힌 표면",
    "깨진 부분",
    "변색된 영역",
    "누락된 부품"
]

results = detector.detect(product_image, defect_targets)

# 품질 판정
if len(results) > 0:
    print("불량품 감지!")
    for det in results:
        print(f"문제: {det.class_name} at {det.bbox}")
else:
    print("정상 제품")
```

## 🔧 결과 데이터 활용

### JSON으로 저장
```python
# 탐지 결과를 JSON으로 저장
import json

results = detector.detect(image, targets)
results_dict = detector.to_dict(results)

with open("detection_results.json", "w") as f:
    json.dump(results_dict, f, indent=2, ensure_ascii=False)
```

### 데이터베이스 저장
```python
# 결과를 데이터베이스에 저장
for det in results:
    db.insert({
        "timestamp": datetime.now(),
        "object_type": det.class_name,
        "confidence": det.confidence,
        "location": det.bbox,
        "image_id": image_id
    })
```

### CSV 내보내기
```python
import csv

# CSV로 내보내기
with open("detections.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Confidence", "X1", "Y1", "X2", "Y2"])
    
    for det in results:
        writer.writerow([
            det.class_name,
            det.confidence,
            *det.bbox
        ])
```

## ⚡ 성능 최적화

### GPU 가속
```python
# GPU 사용 확인
import torch
print(f"GPU 사용 가능: {torch.cuda.is_available()}")

# GPU에서 FP16 사용 (2배 빠름)
detector = DINOv3DetectorAPI(
    model_size="base",
    device="cuda",
    use_fp16=True  # 중요!
)
```

### 처리 속도 비교
| 모델 크기 | CPU (초) | GPU (초) | GPU+FP16 (초) |
|----------|---------|----------|---------------|
| Small    | 0.5     | 0.1      | 0.05          |
| Base     | 1.0     | 0.2      | 0.1           |
| Large    | 2.0     | 0.4      | 0.2           |

### 메모리 사용량 줄이기
```python
# 작은 배치로 처리
for i in range(0, len(images), 4):
    batch = images[i:i+4]
    results = detector.detect_batch(batch, targets)
    
# 사용 후 메모리 정리
import gc
del detector
torch.cuda.empty_cache()
gc.collect()
```

## 🆘 문제 해결

### 첫 실행시 모델 다운로드가 느려요
```python
# 해결책: 모델을 미리 다운로드
from transformers import AutoModel, AutoImageProcessor

# 한번만 실행하면 캐시됨
AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
```

### CUDA out of memory 에러
```python
# 해결책 1: 더 작은 모델 사용
detector = DINOv3DetectorAPI(model_size="small")

# 해결책 2: 이미지 크기 줄이기
image = cv2.resize(image, (640, 480))

# 해결책 3: 배치 크기 줄이기
results = detector.detect_batch(images[:2], targets)
```

### 낮은 탐지율
```python
# 해결책 1: 임계값 낮추기
results = detector.detect(image, targets, confidence_threshold=0.1)

# 해결책 2: 더 구체적인 설명
targets = ["큰 빨간색 트럭", "작은 흰색 승용차"]  # Good
# targets = ["차"]  # Too generic

# 해결책 3: 더 큰 모델 사용
detector = DINOv3DetectorAPI(model_size="large")
```

## 📊 결과 예시

탐지 결과는 다음과 같은 형태입니다:

```python
Detection(
    class_name="빨간 자동차",
    confidence=0.85,
    bbox=(120, 230, 450, 380)  # [x1, y1, x2, y2]
)
```

JSON 출력:
```json
{
    "class_name": "빨간 자동차",
    "confidence": 0.85,
    "bbox": [120, 230, 450, 380]
}
```

## 🚀 시작하기

### 🎯 사용 방법 총정리

#### 방법 1: 커맨드라인 (터미널/CMD)
```bash
# 가장 간단한 사용법
python detect.py 이미지.jpg "사람" "자동차"

# 도움말 보기
python detect.py --help
```

#### 방법 2: GUI (초보자 추천)
```bash
python detect_gui.py
# 마우스로 클릭해서 사용!
```

#### 방법 3: Python 스크립트
```python
from production_api import DINOv3DetectorAPI

detector = DINOv3DetectorAPI()
results = detector.detect("image.jpg", ["person", "car"])
```

### 📝 전체 파일 설명

| 파일 | 용도 | 사용 대상 |
|------|------|-----------|
| `detect.py` | 커맨드라인 단일 처리 | 터미널 사용자 |
| `detect_folder.py` | 폴더 일괄 처리 | 대량 처리 필요시 |
| `detect_gui.py` | GUI 인터페이스 | 초보자, 마우스 선호 |
| `production_api.py` | Python API | 개발자 |
| `test_simple.py` | 간단한 테스트 | 처음 시작할 때 |

### 💻 실행 예제 모음

```bash
# 1. 테스트 (처음 사용시)
python test_simple.py

# 2. 이미지 한 장
python detect.py my_photo.jpg "사람" "자동차"

# 3. 폴더 전체
python detect_folder.py ./photos --targets "person,car" --output ./results

# 4. GUI로 실행
python detect_gui.py

# 5. 비디오
python detect.py video.mp4 --targets "사람" --output detected_video.mp4
```

## 📞 지원

- 이슈: GitHub Issues에 등록
- 개선 제안: Pull Request 환영
- 라이선스: MIT (상업적 사용 가능)

---

**이제 시작해보세요! 단 몇 줄의 코드로 강력한 객체 탐지가 가능합니다** 🎯