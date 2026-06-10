# 오라클 클라우드 무료 티어 배포 가이드

ICT 페이퍼 트레이딩 봇 + 대시보드를 오라클 클라우드 **Always Free** VM에서 24시간 돌리는 방법.
(다른 Linux VM에서도 동일하게 동작 — apt/dnf 자동 감지)

---

## 0. 왜 오라클 무료 티어?

- **ARM Ampere A1**: 최대 4 vCPU + 24GB RAM (이 봇엔 1 vCPU/1GB면 충분)
- **영구 무료** (AWS와 달리 12개월 제한 없음)
- 고정 공인 IP 제공 → 어디서든 대시보드 접속

---

## 1. VM 생성 (오라클 콘솔)

1. [cloud.oracle.com](https://cloud.oracle.com) 로그인 → **Compute → Instances → Create Instance**
2. **Image**: Ubuntu 22.04 (또는 Oracle Linux)
3. **Shape**: `VM.Standard.A1.Flex` (Ampere ARM, Always Free) — 1 OCPU / 6GB 정도면 충분
4. **SSH 키**: 공개키 등록 (또는 키 생성 후 개인키 다운로드)
5. 생성 → **공인 IP** 메모

> A1 ARM 용량이 부족하다고 나오면 `VM.Standard.E2.1.Micro`(AMD, Always Free)도 가능 — 1GB라 빠듯하지만 동작함.

---

## 2. SSH 접속 + 코드 배포

```bash
# 로컬에서 VM 접속 (Ubuntu 이미지는 기본 사용자명 ubuntu)
ssh -i <개인키경로> ubuntu@<공인IP>

# 코드 클론
git clone <이-저장소-URL> TradingBot
cd TradingBot

# 배포 스크립트 실행 (패키지 설치 + 서비스 등록 + 시작)
bash deploy/setup.sh
```

스크립트가 끝나면 봇(`ict-bot`)과 대시보드(`ict-dashboard`)가 systemd 서비스로
등록되어 **자동 시작 + 재부팅 후에도 자동 복구**됩니다.

---

## 3. Discord 알림 (선택)

```bash
echo 'DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...' > ~/TradingBot/.env
sudo systemctl restart ict-bot
```

---

## 4. 대시보드 접속 — 두 가지 방법

### 방법 A: SSH 터널 (권장, 가장 안전)

포트를 외부에 열지 않고, 내 PC에서 암호화 터널로만 접속:

```bash
# 로컬 PC에서 실행
ssh -i <개인키> -L 5000:localhost:5000 ubuntu@<공인IP>
```

→ 브라우저에서 **http://localhost:5000** 접속. VM 포트는 안 열려 있어 외부 노출 0.

> 이 방법을 쓰려면 대시보드를 localhost 바인딩으로 두는 게 좋습니다:
> `sudo DASH_HOST=127.0.0.1 bash deploy/setup.sh` 로 재실행하거나
> 서비스 파일의 `--host` 를 `127.0.0.1` 로 수정 후 `systemctl restart`.

### 방법 B: 포트 직접 개방 (어디서든 브라우저 접속)

1. **오라클 콘솔**: VCN → 서브넷 → **보안 목록** → 인그레스 규칙 추가
   - Source CIDR: `0.0.0.0/0` (또는 내 IP만 `x.x.x.x/32` — 더 안전)
   - IP Protocol: TCP, Destination Port: `5000`
2. **VM 방화벽** (오라클 이미지는 기본 차단):
   ```bash
   sudo iptables -I INPUT 1 -p tcp --dport 5000 -j ACCEPT
   sudo netfilter-persistent save    # Ubuntu, 재부팅 유지
   # Oracle Linux면: sudo firewall-cmd --permanent --add-port=5000/tcp && sudo firewall-cmd --reload
   ```
3. 브라우저: **http://<공인IP>:5000**

> ⚠ 대시보드는 인증이 없는 읽기 전용 화면입니다. 가상 자산이라 위험은 낮지만,
> 공개 노출이 꺼림칙하면 **방법 A(SSH 터널)** 또는 보안목록 Source를 내 IP로 제한하세요.

---

## 5. 운영 명령어

```bash
# 상태 확인
sudo systemctl status ict-bot ict-dashboard

# 실시간 로그
tail -f ~/TradingBot/logs/runner.service.log       # 봇 스캔/진입
tail -f ~/TradingBot/logs/dashboard.service.log    # 대시보드

# 재시작 / 중지
sudo systemctl restart ict-bot
sudo systemctl stop ict-bot ict-dashboard

# 설정 변경 후 반영 (config.yaml 수정 시)
nano ~/TradingBot/config/config.yaml
sudo systemctl restart ict-bot

# 코드 업데이트
cd ~/TradingBot && git pull && sudo systemctl restart ict-bot ict-dashboard
```

---

## 6. 스캔 주기 변경

`config.yaml` 의 `runner.interval_sec` (기본 300초=5분):

```yaml
runner:
  interval_sec: 300   # 60=1분마다, 900=15분마다
```

수정 후 `sudo systemctl restart ict-bot`.

---

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| `setup.sh` 권한 오류 | `bash deploy/setup.sh` (sudo는 내부에서 호출) |
| 대시보드 접속 안 됨 | 보안목록 + iptables 둘 다 열었는지 확인 (4-B) |
| numpy/pandas 설치 느림 | ARM 휠 자동 사용, 첫 설치만 1~2분 소요 |
| 봇이 진입 안 함 | 정상 — 24시간 진입(2026-06 개정)이지만 컨플루언스 70점↑ 셋업만 진입 |
