@echo off
REM ============================================================
REM  ICT Paper Trading Bot - 봇 + 대시보드 동시 실행 (Windows)
REM  창 2개: [봇 러너] + [대시보드]. 끄려면 각 창 닫기.
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0.."

REM --- 실행할 Python 결정 (.venv 있으면 그것, 없으면 시스템) ---
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY="
    where py >nul 2>&1 && set "PY=py"
    if not defined PY ( where python >nul 2>&1 && set "PY=python" )
)
if not defined PY (
    echo [오류] Python을 찾을 수 없습니다. 먼저 windows-setup.bat 실행.
    pause
    exit /b 1
)

REM 봇 러너 (5분 주기 스캔/매매)
start "ICT 봇 러너" cmd /k "%PY% -m src.runner"

REM 대시보드 (외부 접속 허용, 포트 8080)
start "ICT 대시보드" cmd /k "%PY% -m src.dashboard.app --host 0.0.0.0 --port 8080"

echo ============================================================
echo   봇 러너 + 대시보드 시작됨 (창 2개)
echo ============================================================
echo  이 PC에서:          http://127.0.0.1:8080
echo  같은 WiFi 다른 기기: 대시보드 창의 "외부(같은 WiFi)" 줄 IP 사용
echo.
echo  끄려면 두 창을 닫으세요.
echo ============================================================
timeout /t 8 >nul
