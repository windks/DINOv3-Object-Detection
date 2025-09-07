#!/bin/bash

echo "===================================="
echo "DINOv3 Object Detection - Linux/Mac"
echo "===================================="
echo

function show_menu() {
    echo "사용 방법을 선택하세요:"
    echo
    echo "1. GUI 실행 (마우스로 사용)"
    echo "2. 이미지 파일 처리"
    echo "3. 폴더 일괄 처리"
    echo "4. 간단한 테스트"
    echo "5. 종료"
    echo
}

function run_gui() {
    echo
    echo "GUI를 실행합니다..."
    python detect_gui.py
}

function process_image() {
    echo
    read -p "이미지 파일 경로를 입력하세요: " img_path
    read -p "찾을 객체를 입력하세요 (콤마로 구분): " targets
    read -p "결과 저장 경로 (선택사항, Enter로 건너뛰기): " output
    
    if [ -z "$output" ]; then
        python detect.py "$img_path" --targets "$targets"
    else
        python detect.py "$img_path" --targets "$targets" --output "$output"
    fi
}

function process_folder() {
    echo
    read -p "폴더 경로를 입력하세요: " folder_path
    read -p "찾을 객체를 입력하세요 (콤마로 구분): " targets
    read -p "결과 저장 폴더 (선택사항, Enter로 건너뛰기): " output
    
    if [ -z "$output" ]; then
        python detect_folder.py "$folder_path" --targets "$targets"
    else
        python detect_folder.py "$folder_path" --targets "$targets" --output "$output"
    fi
}

function run_test() {
    echo
    echo "간단한 테스트를 실행합니다..."
    python test_simple.py
}

# 메인 루프
while true; do
    show_menu
    read -p "선택 (1-5): " choice
    
    case $choice in
        1) run_gui ;;
        2) process_image ;;
        3) process_folder ;;
        4) run_test ;;
        5) echo; echo "프로그램을 종료합니다."; exit 0 ;;
        *) echo "잘못된 선택입니다. 다시 선택하세요." ;;
    esac
    
    echo
    read -p "계속하려면 Enter를 누르세요..."
    clear
done