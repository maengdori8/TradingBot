#!/bin/bash
# Oracle Cloud Always Free 인스턴스 초기 설정 스크립트
# 사용법: ssh ubuntu@<서버IP> 접속 후
#   curl -fsSL https://raw.githubusercontent.com/<user>/TradingBot/main/deploy/setup.sh | bash
# 또는 파일 복사 후: chmod +x setup.sh && ./setup.sh

set -e

echo "============================================"
echo "  ICT Trading Bot — 서버 설정 시작"
echo "============================================"

# 1. 시스템 패키지 업데이트
echo "[1/6] 시스템 업데이트..."
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Docker 설치
echo "[2/6] Docker 설치..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "  Docker 설치 완료 (재로그인 필요할 수 있음)"
else
    echo "  Docker 이미 설치됨"
fi

# 3. Docker Compose 설치
echo "[3/6] Docker Compose 확인..."
if ! docker compose version &> /dev/null; then
    sudo apt-get install -y docker-compose-plugin
fi

# 4. 프로젝트 클론
echo "[4/6] 프로젝트 설정..."
BOT_DIR="$HOME/TradingBot"
if [ ! -d "$BOT_DIR" ]; then
    echo "  TradingBot 디렉토리가 없습니다."
    echo "  git clone <your-repo> ~/TradingBot 후 이 스크립트를 다시 실행하세요."
    echo "  또는 scp로 파일을 복사하세요."
    mkdir -p "$BOT_DIR"
fi

cd "$BOT_DIR"

# 5. .env 파일 확인
echo "[5/6] 환경변수 설정..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "  .env 파일 생성됨 — API 키를 입력하세요:"
        echo "    nano ~/TradingBot/.env"
    else
        echo "  .env.example 파일이 없습니다."
    fi
else
    echo "  .env 파일 존재함"
fi

# 6. 로그 디렉토리 생성
mkdir -p logs paper_logs

# 7. 방화벽 설정 (Oracle Cloud iptables)
echo "[6/6] 방화벽 확인..."
sudo iptables -L INPUT -n | grep -q "8080" || true

echo ""
echo "============================================"
echo "  설정 완료!"
echo "============================================"
echo ""
echo "  다음 단계:"
echo ""
echo "  1. API 키 설정:"
echo "     nano ~/TradingBot/.env"
echo ""
echo "  2. 설정 조정 (심볼, 타임프레임 등):"
echo "     nano ~/TradingBot/config/settings.yaml"
echo ""
echo "  3. Paper 모드로 실행:"
echo "     cd ~/TradingBot"
echo "     docker compose up -d"
echo ""
echo "  4. 로그 확인:"
echo "     docker compose logs -f"
echo ""
echo "  5. 중지:"
echo "     docker compose down"
echo ""
echo "============================================"
