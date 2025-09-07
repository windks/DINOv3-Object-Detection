#!/usr/bin/env python3
"""
폴더 내 모든 이미지를 일괄 처리하는 스크립트

사용법:
    python detect_folder.py ./images "사람" "자동차" --output ./results
    python detect_folder.py ./photos --targets "person,car,dog" --csv results.csv
"""

import argparse
from pathlib import Path
import cv2
import csv
import json
from datetime import datetime
from tqdm import tqdm
import os

from production_api import DINOv3DetectorAPI


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="폴더 내 모든 이미지에서 객체 탐지",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 사용
  python detect_folder.py ./images "사람" "자동차"
  
  # 결과 폴더 지정
  python detect_folder.py ./images --targets "person,car" --output ./results
  
  # CSV로 결과 저장
  python detect_folder.py ./photos --targets "사람,개,고양이" --csv detections.csv
  
  # 재귀적으로 하위 폴더 포함
  python detect_folder.py ./dataset --targets "vehicle" --recursive
        """
    )
    
    parser.add_argument(
        "folder",
        help="처리할 이미지 폴더 경로"
    )
    
    parser.add_argument(
        "targets",
        nargs="*",
        help="찾을 객체들"
    )
    
    parser.add_argument(
        "--targets",
        "-t",
        dest="targets_comma",
        help="콤마로 구분된 타겟"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="결과 이미지 저장 폴더"
    )
    
    parser.add_argument(
        "--csv",
        help="CSV 결과 파일 경로"
    )
    
    parser.add_argument(
        "--json",
        help="JSON 결과 파일 경로"
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
        help="모델 크기"
    )
    
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="하위 폴더 포함"
    )
    
    parser.add_argument(
        "--extensions",
        "-e",
        default="jpg,jpeg,png,bmp",
        help="처리할 이미지 확장자 (기본: jpg,jpeg,png,bmp)"
    )
    
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="객체가 없는 이미지는 저장 안함"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="요약 정보만 출력"
    )
    
    return parser.parse_args()


def find_images(folder_path, extensions, recursive):
    """폴더에서 이미지 파일 찾기"""
    folder = Path(folder_path)
    image_files = []
    
    ext_list = [f".{e.strip()}" for e in extensions.split(",")]
    
    if recursive:
        for ext in ext_list:
            image_files.extend(folder.rglob(f"*{ext}"))
            image_files.extend(folder.rglob(f"*{ext.upper()}"))
    else:
        for ext in ext_list:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    return sorted(set(image_files))


def process_batch(detector, image_files, targets, args):
    """이미지 일괄 처리"""
    results_all = []
    summary = {
        "total_images": len(image_files),
        "images_with_detections": 0,
        "total_detections": 0,
        "detections_by_class": {}
    }
    
    # 출력 폴더 생성
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # CSV 파일 준비
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image', 'Object', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
    
    # 진행 표시
    print(f"\n🔍 {len(image_files)}개 이미지 처리 중...")
    print(f"찾는 객체: {targets}\n")
    
    # 각 이미지 처리
    for img_path in tqdm(image_files, desc="처리 진행"):
        # 이미지 로드
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️  읽기 실패: {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 탐지
        results = detector.detect(
            image_rgb,
            targets,
            confidence_threshold=args.threshold
        )
        
        # 통계 업데이트
        if results:
            summary["images_with_detections"] += 1
            summary["total_detections"] += len(results)
            
            for det in results:
                if det.class_name in summary["detections_by_class"]:
                    summary["detections_by_class"][det.class_name] += 1
                else:
                    summary["detections_by_class"][det.class_name] = 1
        
        # 결과 저장
        result_data = {
            "image": str(img_path),
            "detections": results,
            "count": len(results)
        }
        results_all.append(result_data)
        
        # CSV 저장
        if csv_writer:
            for det in results:
                csv_writer.writerow([
                    img_path.name,
                    det.class_name,
                    f"{det.confidence:.3f}",
                    *det.bbox
                ])
        
        # 이미지 저장
        if args.output and (results or not args.skip_empty):
            vis_image = detector.visualize(image_rgb, results)
            output_file = output_path / f"{img_path.stem}_detected{img_path.suffix}"
            cv2.imwrite(str(output_file), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # 상세 출력 (summary 모드가 아닐 때)
        if not args.summary and results:
            print(f"\n📄 {img_path.name}: {len(results)}개 탐지")
            for det in results:
                print(f"   - {det.class_name} ({det.confidence:.1%})")
    
    # CSV 파일 닫기
    if csv_file:
        csv_file.close()
        print(f"\n📊 CSV 저장 완료: {args.csv}")
    
    # JSON 저장
    if args.json:
        save_batch_json(results_all, args.json, summary)
    
    return summary


def save_batch_json(results_all, json_path, summary):
    """일괄 처리 결과를 JSON으로 저장"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": []
    }
    
    for result in results_all:
        data["results"].append({
            "image": result["image"],
            "count": result["count"],
            "detections": [
                {
                    "class_name": det.class_name,
                    "confidence": float(det.confidence),
                    "bbox": list(det.bbox)
                }
                for det in result["detections"]
            ]
        })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 JSON 저장 완료: {json_path}")


def print_summary(summary):
    """요약 정보 출력"""
    print("\n" + "=" * 60)
    print("📊 처리 요약")
    print("=" * 60)
    print(f"전체 이미지: {summary['total_images']}개")
    print(f"객체가 탐지된 이미지: {summary['images_with_detections']}개 "
          f"({summary['images_with_detections']/summary['total_images']*100:.1f}%)")
    print(f"전체 탐지 수: {summary['total_detections']}개")
    
    if summary['detections_by_class']:
        print("\n클래스별 탐지 수:")
        for class_name, count in sorted(summary['detections_by_class'].items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"  - {class_name}: {count}개")
    
    print("=" * 60)


def main():
    args = parse_arguments()
    
    # 타겟 구성
    targets = []
    if args.targets:
        targets.extend(args.targets)
    if args.targets_comma:
        targets.extend([t.strip() for t in args.targets_comma.split(",")])
    
    if not targets:
        print("❌ 오류: 찾을 객체를 지정하세요!")
        return
    
    # 폴더 확인
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        return
    
    # 이미지 파일 찾기
    image_files = find_images(folder_path, args.extensions, args.recursive)
    
    if not image_files:
        print(f"⚠️  이미지를 찾을 수 없습니다: {folder_path}")
        return
    
    print(f"\n🚀 DINOv3 일괄 처리 시작")
    print(f"📂 폴더: {folder_path}")
    print(f"🖼️  찾은 이미지: {len(image_files)}개")
    
    # 디텍터 초기화
    print(f"\n📦 모델 로딩 중... (model={args.model})")
    detector = DINOv3DetectorAPI(model_size=args.model)
    print("✅ 모델 로딩 완료!")
    
    # 일괄 처리
    summary = process_batch(detector, image_files, targets, args)
    
    # 요약 출력
    print_summary(summary)
    
    if args.output:
        print(f"\n✅ 결과 이미지 저장: {args.output}")
    
    print("\n✨ 일괄 처리 완료!")


if __name__ == "__main__":
    main()