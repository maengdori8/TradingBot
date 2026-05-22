#!/bin/bash
# 자주 쓰는 서버 관리 명령어 모음

BOT_DIR="$HOME/TradingBot"
cd "$BOT_DIR"

case "$1" in
    start)
        echo "봇 시작 (paper 모드)..."
        docker compose up -d
        docker compose logs -f
        ;;
    stop)
        echo "봇 중지..."
        docker compose down
        ;;
    restart)
        echo "봇 재시작..."
        docker compose restart
        docker compose logs -f
        ;;
    logs)
        docker compose logs -f --tail=100
        ;;
    status)
        docker compose ps
        echo ""
        if [ -f healthcheck ]; then
            echo "헬스체크:"
            cat healthcheck
        fi
        ;;
    update)
        echo "코드 업데이트 및 재빌드..."
        git pull
        docker compose down
        docker compose build --no-cache
        docker compose up -d
        echo "업데이트 완료"
        docker compose logs -f
        ;;
    paper-report)
        echo "최근 페이퍼 트레이딩 리포트:"
        ls -t paper_logs/report_*.txt 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "리포트 없음"
        ;;
    trades)
        echo "최근 거래 기록:"
        ls -t paper_logs/trades_*.csv 2>/dev/null | head -1 | xargs tail -20 2>/dev/null || echo "거래 기록 없음"
        ;;
    *)
        echo "사용법: ./commands.sh {start|stop|restart|logs|status|update|paper-report|trades}"
        ;;
esac
