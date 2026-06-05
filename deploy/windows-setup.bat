@echo off
REM ============================================================
REM  ICT Paper Trading Bot - Windows 최초 설치 (한 번만 실행)
REM  사용법: 이 파일을 더블클릭하거나 cmd에서 실행
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0\.."

echo ============================================================
echo   ICT Paper Trading Bot - Windows 설치
echo ============================================================
echo  경로: %CD%
echo.

REM Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 없습니다. https://www.python.org/downloads/ 에서
    echo        설치 시 "Add Python to PATH" 체크 후 다시 실행하세요.
    pause
    exit /b 1
)

echo [1/2] 가상환경 생성...
if not exist ".venv" (
    python -m venv .venv
)

echo [2/2] 패키지 설치... (numpy/pandas 첫 설치는 1~2분 소요)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

if not exist "logs" mkdir logs

echo.
echo ============================================================
echo   설치 완료!
echo ============================================================
echo  Discord 알림을 쓰려면 프로젝트 폴더에 .env 파일을 만들고:
echo      DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
echo.
echo  이제 windows-run.bat 을 더블클릭하면 봇+대시보드가 켜집니다.
echo ============================================================
pause
