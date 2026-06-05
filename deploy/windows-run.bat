@echo off
REM ============================================================
REM  ICT Paper Trading Bot - 봇 + 대시보드 동시 실행 (Windows)
REM  창 2개가 뜹니다: [봇 러너] + [대시보드]. 끄려면 각 창 닫기.
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo [오류] 먼저 windows-setup.bat 을 실행해 설치하세요.
    pause
    exit /b 1
)

REM 봇 러너 (5분 주기 스캔/매매)
start "ICT 봇 러너" cmd /k ".venv\Scripts\python -m src.runner"

REM 대시보드 (외부 접속 허용, 포트 8080)
start "ICT 대시보드" cmd /k ".venv\Scripts\python -m src.dashboard.app --host 0.0.0.0 --port 8080"

echo ============================================================
echo   봇 러너 + 대시보드 시작됨 (창 2개)
echo ============================================================
echo  이 PC에서:        http://127.0.0.1:8080
echo  같은 WiFi 다른기기: 대시보드 창의 "외부(같은 WiFi)" 줄 IP 사용
echo.
echo  끄려면 두 창을 닫으세요.
echo ============================================================
timeout /t 8 >nul
