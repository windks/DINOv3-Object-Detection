#!/usr/bin/env python3
"""
DINOv3 Object Detection - 로컬 실행 스크립트
커맨드라인에서 직접 실행 가능한 독립형 프로그램

사용법:
    python detect.py image.jpg "사람" "자동차" "강아지"
    python detect.py image.jpg --targets "사람,자동차,강아지" --output result.jpg
    python detect.py video.mp4 --targets "person,car" --output output.mp4
"""

import argparse
import cv2
import sys
import os
from pathlib import Path
from typing import List
import json

from production_api import DINOv3DetectorAPI, process_video


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DINOv3 Object Detection - 로컬 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 이미지에서 객체 찾기
  python detect.py photo.jpg "사람" "자동차" "강아지"
  
  # 콤마로 구분된 타겟
  python detect.py photo.jpg --targets "person,car,dog"
  
  # 결과 저장
  python detect.py photo.jpg --targets "사람,자동차" --output detected.jpg
  
  # 비디오 처리
  python detect.py video.mp4 --targets "person,car" --output result.mp4
  
  # 상세 옵션
  python detect.py photo.jpg --targets "빨간 차,파란 차" --threshold 0.2 --model large
        """
    )
    
    parser.add_argument(
        "input",
        help="입력 이미지 또는 비디오 파일 경로"
    )
    
    parser.add_argument(
        "targets",
        nargs="*",
        help="찾을 객체들 (예: 사람 자동차 강아지)"
    )
    
    parser.add_argument(
        "--targets",
        "-t",
        dest="targets_comma",
        help="콤마로 구분된 타겟 리스트 (예: 사람,자동차,강아지)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="결과 저장 경로 (이미지/비디오)"
    )
    
    parser.add_argument(
        "--threshold",
        "-th",
        type=float,
        default=0.3,
        help="신뢰도 임계값 (기본: 0.3)"
    )
    
    parser.add_argument(
        "--model",
        "-m",
        choices=["small", "base", "large"],
        default="small",
        help="모델 크기 (기본: small)"
    )
    
    parser.add_argument(
        "--json",
        "-j",
        help="JSON 결과 저장 경로"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="결과 화면에 표시 안함"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="GPU 사용 (사용 가능한 경우)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="탐지된 객체 목록만 출력"
    )
    
    return parser.parse_args()


def process_image(detector, image_path, targets, args):
    """이미지 처리"""
    print(f"\n🖼️  이미지 처리 중: {image_path}")
    
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 이미지를 열 수 없습니다: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 탐지
    print(f"🔍 찾는 객체: {targets}")
    results = detector.detect(
        image_rgb,
        targets,
        confidence_threshold=args.threshold
    )
    
    # 결과 출력
    print(f"\n📊 탐지 결과: {len(results)}개 객체 발견")
    if results:
        print("-" * 50)
        for i, det in enumerate(results, 1):
            print(f"{i}. {det.class_name}")
            print(f"   신뢰도: {det.confidence:.1%}")
            print(f"   위치: {det.bbox}")
        print("-" * 50)
    
    # 리스트만 출력 모드
    if args.list:
        detected_names = [det.class_name for det in results]
        print("\n탐지된 객체:", ", ".join(detected_names))
    
    # JSON 저장
    if args.json:
        save_json(results, args.json, image_path)
    
    # 시각화
    if args.output or not args.no_display:
        vis_image = detector.visualize(image_rgb, results)
        
        if args.output:
            output_path = args.output if args.output.endswith(('.jpg', '.png')) else f"{args.output}.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"✅ 결과 저장: {output_path}")
        
        if not args.no_display:
            # 화면에 표시
            cv2.imshow('Detection Result', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print("\n아무 키나 누르면 종료됩니다...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def process_video_file(detector, video_path, targets, args):
    """비디오 처리"""
    print(f"\n🎬 비디오 처리 중: {video_path}")
    
    if not args.output:
        output_path = video_path.stem + "_detected.mp4"
    else:
        output_path = args.output
    
    print(f"🔍 찾는 객체: {targets}")
    print(f"📹 출력 비디오: {output_path}")
    
    # 비디오 처리
    process_video(
        str(video_path),
        str(output_path),
        targets,
        threshold=args.threshold
    )
    
    print(f"✅ 비디오 처리 완료: {output_path}")


def save_json(results, json_path, source_path):
    """JSON으로 결과 저장"""
    data = {
        "source": str(source_path),
        "detections": [
            {
                "class_name": det.class_name,
                "confidence": float(det.confidence),
                "bbox": list(det.bbox)
            }
            for det in results
        ],
        "total": len(results)
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 JSON 저장: {json_path}")


def main():
    # 명령줄 인자 파싱
    args = parse_arguments()
    
    # 타겟 리스트 구성
    targets = []
    if args.targets:
        targets.extend(args.targets)
    if args.targets_comma:
        targets.extend([t.strip() for t in args.targets_comma.split(",")])
    
    if not targets:
        print("❌ 오류: 찾을 객체를 지정하세요!")
        print("예: python detect.py image.jpg 사람 자동차")
        print("또는: python detect.py image.jpg --targets 사람,자동차")
        sys.exit(1)
    
    # 입력 파일 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    # 디텍터 초기화
    print(f"\n🚀 DINOv3 Object Detection 시작")
    print(f"📦 모델 로딩 중... (model={args.model})")
    
    device = "cuda" if args.gpu else None
    detector = DINOv3DetectorAPI(
        model_size=args.model,
        device=device,
        use_fp16=args.gpu
    )
    print("✅ 모델 로딩 완료!")
    
    # 파일 타입에 따라 처리
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # 이미지 처리
        process_image(detector, input_path, targets, args)
        
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # 비디오 처리
        process_video_file(detector, input_path, targets, args)
        
    else:
        print(f"❌ 지원하지 않는 파일 형식: {input_path.suffix}")
        sys.exit(1)
    
    print("\n✨ 완료!")


if __name__ == "__main__":
    main()