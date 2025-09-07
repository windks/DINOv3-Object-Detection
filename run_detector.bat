@echo off
echo ====================================
echo DINOv3 Object Detection - Windows
echo ====================================
echo.

:menu
echo 사용 방법을 선택하세요:
echo.
echo 1. GUI 실행 (마우스로 사용)
echo 2. 이미지 파일 처리
echo 3. 폴더 일괄 처리
echo 4. 간단한 테스트
echo 5. 종료
echo.

set /p choice="선택 (1-5): "

if "%choice%"=="1" goto gui
if "%choice%"=="2" goto image
if "%choice%"=="3" goto folder
if "%choice%"=="4" goto test
if "%choice%"=="5" goto end

echo 잘못된 선택입니다. 다시 선택하세요.
goto menu

:gui
echo.
echo GUI를 실행합니다...
python detect_gui.py
pause
goto menu

:image
echo.
set /p img_path="이미지 파일 경로를 입력하세요: "
set /p targets="찾을 객체를 입력하세요 (콤마로 구분): "
set /p output="결과 저장 경로 (선택사항, Enter로 건너뛰기): "

if "%output%"=="" (
    python detect.py "%img_path%" --targets "%targets%"
) else (
    python detect.py "%img_path%" --targets "%targets%" --output "%output%"
)
pause
goto menu

:folder
echo.
set /p folder_path="폴더 경로를 입력하세요: "
set /p targets="찾을 객체를 입력하세요 (콤마로 구분): "
set /p output="결과 저장 폴더 (선택사항, Enter로 건너뛰기): "

if "%output%"=="" (
    python detect_folder.py "%folder_path%" --targets "%targets%"
) else (
    python detect_folder.py "%folder_path%" --targets "%targets%" --output "%output%"
)
pause
goto menu

:test
echo.
echo 간단한 테스트를 실행합니다...
python test_simple.py
pause
goto menu

:end
echo.
echo 프로그램을 종료합니다.
pause