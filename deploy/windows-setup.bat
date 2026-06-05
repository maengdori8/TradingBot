@echo off
REM ============================================================
REM  ICT Paper Trading Bot - Windows 최초 설치 (한 번만 실행)
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0.."

echo ============================================================
echo   ICT Paper Trading Bot - Windows 설치
echo ============================================================
echo  경로: %CD%
echo.

REM --- Python 런처 탐지 (py 우선, 없으면 python) ---
set "PYCMD="
where py >nul 2>&1 && set "PYCMD=py"
if not defined PYCMD ( where python >nul 2>&1 && set "PYCMD=python" )
if not defined PYCMD (
    echo [오류] Python을 찾을 수 없습니다.
    echo        https://www.python.org/downloads/ 에서 설치 시
    echo        "Add python.exe to PATH" 체크 후 다시 실행하세요.
    pause
    exit /b 1
)
echo 사용할 Python: %PYCMD%
%PYCMD% --version
echo.

REM --- 가상환경 시도 ---
echo [1/2] 가상환경(.venv) 생성 시도...
%PYCMD% -m venv .venv

if exist ".venv\Scripts\python.exe" (
    set "PIP=.venv\Scripts\python.exe"
    echo   가상환경 생성 성공.
) else (
    echo   [경고] 가상환경 생성 실패 - 일반 설치로 대체합니다.
    echo          ^(Microsoft Store 버전 Python일 가능성. 정식판 권장^)
    set "PIP=%PYCMD%"
)
echo.

REM --- 패키지 설치 ---
echo [2/2] 패키지 설치... (numpy/pandas 첫 설치는 1~2분 소요)
%PIP% -m pip install --upgrade pip
%PIP% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [오류] 패키지 설치 실패. 위 메시지를 확인하세요.
    pause
    exit /b 1
)

if not exist "logs" mkdir logs

echo.
echo ============================================================
echo   설치 완료!
echo ============================================================
echo  Discord 알림: 프로젝트 폴더에 .env 파일 생성 후
echo      DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
echo.
echo  이제 windows-run.bat 을 더블클릭하세요.
echo ============================================================
pause
