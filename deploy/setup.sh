#!/usr/bin/env bash
#
# ICT Paper Trading Bot — 오라클 클라우드(또는 임의의 Linux VM) 배포 스크립트
#
# 사용법:
#   git clone <repo> && cd TradingBot
#   bash deploy/setup.sh
#
# 동작:
#   1) 시스템 패키지(python venv/pip/빌드도구) 설치 (apt 또는 dnf 자동 감지)
#   2) .venv 가상환경 생성 + requirements 설치
#   3) systemd 서비스 2개 등록: ict-bot(주기 실행), ict-dashboard(웹 UI)
#   4) 서비스 활성화 + 시작 (재부팅 후 자동 시작)
#
set -euo pipefail

# ── 경로/사용자 자동 감지 ────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="$(whoami)"
VENV_DIR="$REPO_DIR/.venv"
PY="$VENV_DIR/bin/python"
DASH_PORT="${DASH_PORT:-5000}"
DASH_HOST="${DASH_HOST:-0.0.0.0}"

echo "════════════════════════════════════════════════════"
echo "  ICT Paper Trading Bot 배포"
echo "════════════════════════════════════════════════════"
echo "  저장소 경로 : $REPO_DIR"
echo "  실행 사용자 : $RUN_USER"
echo "  대시보드    : http://${DASH_HOST}:${DASH_PORT}"
echo "════════════════════════════════════════════════════"

# ── 1) 시스템 패키지 설치 ────────────────────────────────────────────
echo "[1/4] 시스템 패키지 설치..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-venv python3-pip python3-dev build-essential git
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ make git
else
    echo "  ⚠ apt/dnf 둘 다 없음 — 수동으로 python3/pip/venv 설치 필요"
fi

# ── 2) 가상환경 + 의존성 ─────────────────────────────────────────────
echo "[2/4] 가상환경 생성 및 의존성 설치..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$REPO_DIR/requirements.txt"

mkdir -p "$REPO_DIR/logs"

# .env 안내
if [ ! -f "$REPO_DIR/.env" ]; then
    echo "  ℹ .env 파일이 없습니다. Discord 알림을 쓰려면 생성하세요:"
    echo "      echo 'DISCORD_WEBHOOK_URL=...' > $REPO_DIR/.env"
fi

# ── 3) systemd 서비스 등록 ───────────────────────────────────────────
echo "[3/4] systemd 서비스 등록..."

sudo tee /etc/systemd/system/ict-bot.service >/dev/null <<EOF
[Unit]
Description=ICT Paper Trading Bot (주기 스캔/진입 러너)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$REPO_DIR
ExecStart=$PY -m src.runner
Restart=always
RestartSec=10
StandardOutput=append:$REPO_DIR/logs/runner.service.log
StandardError=append:$REPO_DIR/logs/runner.service.log

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/ict-dashboard.service >/dev/null <<EOF
[Unit]
Description=ICT Paper Trading Dashboard (Flask 웹 UI)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$REPO_DIR
ExecStart=$PY -m src.dashboard.app --host $DASH_HOST --port $DASH_PORT
Restart=always
RestartSec=10
StandardOutput=append:$REPO_DIR/logs/dashboard.service.log
StandardError=append:$REPO_DIR/logs/dashboard.service.log

[Install]
WantedBy=multi-user.target
EOF

# ── 4) 서비스 시작 ───────────────────────────────────────────────────
echo "[4/4] 서비스 활성화 및 시작..."
sudo systemctl daemon-reload
sudo systemctl enable ict-bot ict-dashboard
sudo systemctl restart ict-bot ict-dashboard

echo ""
echo "════════════════════════════════════════════════════"
echo "  ✅ 배포 완료"
echo "════════════════════════════════════════════════════"
echo "  상태 확인 : sudo systemctl status ict-bot ict-dashboard"
echo "  봇 로그   : tail -f $REPO_DIR/logs/runner.service.log"
echo "  대시 로그 : tail -f $REPO_DIR/logs/dashboard.service.log"
echo ""
echo "  ⚠ 대시보드 외부 접속하려면 포트 $DASH_PORT 개방 필요:"
echo "     1) 오라클 콘솔: VCN → 보안 목록 → 인그레스 규칙 추가 (TCP $DASH_PORT)"
echo "     2) VM 방화벽:   sudo iptables -I INPUT 1 -p tcp --dport $DASH_PORT -j ACCEPT"
echo "                     (또는 deploy/README.md의 SSH 터널 방식 권장)"
echo "════════════════════════════════════════════════════"
