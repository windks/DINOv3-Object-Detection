"""
가장 간단한 테스트 스크립트
이 파일을 실행해서 바로 테스트해보세요!
"""

from production_api import DINOv3DetectorAPI
import numpy as np
import cv2

print("DINOv3 Object Detection 테스트")
print("-" * 50)

# 1. 탐지기 초기화 (첫 실행시 모델 다운로드 - 시간이 걸립니다)
print("1. 모델 로딩중... (첫 실행시 다운로드로 1-2분 소요)")
detector = DINOv3DetectorAPI(model_size="small")  # 빠른 테스트용
print("✓ 모델 로딩 완료!")

# 2. 테스트 이미지 생성 (실제로는 cv2.imread("your_image.jpg") 사용)
print("\n2. 테스트 이미지 생성중...")
# 간단한 테스트 이미지 생성
test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255  # 흰 배경

# 빨간 사각형 (자동차처럼 보이게)
cv2.rectangle(test_image, (100, 300), (300, 450), (0, 0, 255), -1)
cv2.rectangle(test_image, (120, 280), (280, 320), (0, 0, 200), -1)  # 창문

# 파란 원 (사람 머리처럼)
cv2.circle(test_image, (500, 200), 50, (255, 0, 0), -1)
cv2.rectangle(test_image, (450, 250), (550, 450), (255, 0, 0), -1)  # 몸통

# 녹색 타원 (강아지처럼)
cv2.ellipse(test_image, (650, 400), (80, 50), 0, 0, 360, (0, 255, 0), -1)

cv2.imwrite("test_input.jpg", test_image)
print("✓ 테스트 이미지 생성 완료! (test_input.jpg)")

# 3. 객체 탐지
print("\n3. 객체 탐지 중...")
targets = [
    "빨간색 자동차",
    "파란색 사람",
    "초록색 동물",
    "노란색 버스",
    "검은색 자전거"
]

print(f"찾는 객체: {targets}")

# RGB로 변환 (OpenCV는 BGR 사용)
test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# 탐지 실행!
results = detector.detect(test_image_rgb, targets, confidence_threshold=0.2)

# 4. 결과 출력
print(f"\n4. 탐지 결과: {len(results)}개 객체 발견")
print("-" * 50)

if len(results) > 0:
    for i, det in enumerate(results, 1):
        print(f"{i}. {det.class_name}")
        print(f"   - 신뢰도: {det.confidence:.1%}")
        print(f"   - 위치: {det.bbox}")
        print()
    
    # 5. 시각화
    print("5. 결과 시각화 중...")
    vis_image = detector.visualize(test_image_rgb, results)
    cv2.imwrite("test_output.jpg", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print("✓ 결과 이미지 저장 완료! (test_output.jpg)")
else:
    print("객체를 찾지 못했습니다.")
    print("팁: confidence_threshold를 낮춰보세요 (현재: 0.2)")

print("\n" + "=" * 50)
print("테스트 완료!")
print("=" * 50)

# 실제 이미지로 테스트하는 방법
print("\n실제 이미지로 테스트하려면:")
print("""
# 1. 이미지 파일 준비 (예: photo.jpg)
# 2. 아래 코드 실행:

image = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = detector.detect(
    image_rgb,
    ["사람", "자동차", "강아지", "고양이"],  # 찾고 싶은 것
    confidence_threshold=0.3
)

# 시각화
vis = detector.visualize(image_rgb, results, "my_result.jpg")
""")