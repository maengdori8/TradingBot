# Google Cloud 무료 티어 배포 가이드

오라클 대안. **e2-micro 영구 무료(Always Free) VM**에서 봇 + 대시보드 24시간 운영.
장점: **브라우저 SSH** (키 설정 불필요), 한국 카드 통과율이 오라클보다 높음.

---

## ⚠️ 영구 무료 조건 (꼭 지킬 것)

Always Free는 **아래 조건을 정확히** 맞춰야 과금 안 됨:

- 머신 타입: **`e2-micro`** (반드시 이것)
- 리전: **`us-west1`(오레곤) / `us-central1`(아이오와) / `us-east1`(사우스캐롤라이나)** 중 하나만
- 부팅 디스크: **표준(Standard) 30GB 이하**
- e2-micro **1대만**

> 가입 시 $300/90일 크레딧도 주지만, 위 조건이면 크레딧 안 쓰고 영구 무료.
> 미국 리전이라 한국에서 SSH/대시보드 접속이 ~0.2초 느리지만, 5분 봇 루프엔 무관.

---

## 1. 가입 + VM 생성

1. [console.cloud.google.com](https://console.cloud.google.com) 가입 (카드 본인확인)
2. 상단 검색 → **Compute Engine** → 사용 설정(Enable)
3. **인스턴스 만들기**:
   - 이름: `ict-bot`
   - 리전: **us-central1 (Iowa)**, 영역(zone): 아무거나
   - 머신 구성: **E2** 시리즈 → **e2-micro** (선택 시 "월 $0 예상" 확인)
   - 부팅 디스크: **변경** → Ubuntu 22.04 LTS, 표준 30GB
   - 방화벽: **HTTP/HTTPS 허용 체크 불필요** (포트 5000은 아래서 따로)
   - **만들기**

---

## 2. 방화벽 — 대시보드 포트 5000 열기

좌측 메뉴 → **VPC 네트워크 → 방화벽 → 방화벽 규칙 만들기**:

- 이름: `allow-dashboard`
- 방향: 인그레스 / 대상: 네트워크의 모든 인스턴스
- 소스 IPv4 범위: `0.0.0.0/0` (또는 내 IP만 — 더 안전)
- 프로토콜/포트: **TCP** 체크 → `5000`
- 만들기

> 더 안전하게 가려면 이 단계 건너뛰고 SSH 터널만 써도 됨 (아래 5번 참고).

---

## 3. SSH 접속 (키 불필요!)

인스턴스 목록에서 `ict-bot` 줄의 **`SSH` 버튼 클릭** → 브라우저에 터미널 창이 뜸. 끝.

---

## 4. 배포 (브라우저 SSH 안에서)

```bash
# 코드 가져오기 (저장소가 public이면 그대로, private면 토큰 필요)
git clone https://github.com/maengdori8/TradingBot.git
cd TradingBot

# 배포 — 패키지/스왑/서비스 자동 설정
bash deploy/setup.sh
```

스크립트가 자동으로: 패키지 설치 → (1GB RAM이라) **2GB 스왑 생성** → 가상환경 →
systemd 서비스 2개 등록 + 시작. 재부팅해도 자동 복구.

Discord 알림:
```bash
echo 'DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...' > ~/TradingBot/.env
sudo systemctl restart ict-bot
```

---

## 5. 대시보드 접속

- **포트 열었으면**: 브라우저 → `http://<VM외부IP>:5000` (외부 IP는 인스턴스 목록에 표시)
- **SSH 터널(더 안전)**: gcloud CLI 또는 콘솔 SSH로는 번거로우니, 포트 개방 방식 권장

---

## 6. 운영

```bash
sudo systemctl status ict-bot ict-dashboard      # 상태
tail -f ~/TradingBot/logs/runner.service.log     # 봇 로그
cd ~/TradingBot && git pull && sudo systemctl restart ict-bot ict-dashboard  # 업데이트
```

---

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| VM 생성 시 과금 경고 | e2-micro + 미국 리전 + 표준 30GB 맞는지 재확인 |
| 메모리 부족으로 느림/멈춤 | setup.sh가 스왑 자동 생성함. `free -h`로 확인 |
| `git clone` 인증 요구 | 저장소가 private → GitHub 토큰 필요 (아래) |
| 대시보드 접속 안 됨 | 방화벽 규칙(2번) + 포트 5000 확인 |
