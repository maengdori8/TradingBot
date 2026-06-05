# Windows PC 운영 가이드

맥은 개발용, **실제 봇은 Windows PC에서 24시간** 돌리는 구성.
배치 파일 더블클릭만으로 설치·실행되도록 만들어 두었다.

---

## 0. 먼저 — 코드를 PC로 가져오기

PC에 최신 코드가 있어야 한다. 두 방법 중 하나:

### 방법 A: GitHub (권장)
맥에서 한 번 push 해두면 PC에서 clone/pull 로 항상 최신 동기화.

```bash
# (맥) 최초 1회 push — GitHub 토큰 필요
cd /Users/m/Documents/GitHub/TradingBot
git push origin main
```
> Username: `maengdori8` / Password: GitHub 토큰
> (https://github.com/settings/tokens/new → repo 체크 → 생성)

```powershell
:: (PC) 코드 받기
git clone https://github.com/maengdori8/TradingBot.git
cd TradingBot
```
이후 업데이트는 `git pull` 한 줄.

### 방법 B: 폴더 복사
GitHub 안 쓸 거면 맥의 `TradingBot` 폴더를 통째로 USB/네트워크로 PC에 복사.
단 `.venv`, `logs`, `__pycache__` 폴더는 빼고 복사할 것 (PC에서 새로 생성됨).

---

## 1. Python 설치 (최초 1회)

https://www.python.org/downloads/ → 3.11+ 설치
**설치 첫 화면에서 "Add python.exe to PATH" 체크 필수.**

---

## 2. 설치 (최초 1회)

`deploy\windows-setup.bat` **더블클릭**.
→ 가상환경 생성 + 패키지 설치 자동. (numpy/pandas 첫 설치 1~2분)

Discord 알림 쓰려면 프로젝트 폴더에 `.env` 파일 생성:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

---

## 3. 실행

`deploy\windows-run.bat` **더블클릭**.
→ 창 2개가 뜬다: **[봇 러너]** (5분마다 스캔/매매) + **[대시보드]**

- 이 PC에서 보기: **http://127.0.0.1:8080**
- 같은 WiFi 다른 기기(폰 등): 대시보드 창에 출력되는 `외부(같은 WiFi): http://192.168.x.x:8080` 주소 사용

끄려면 두 창을 닫으면 된다.

> **첫 실행 시 Windows Defender 방화벽 팝업**이 뜨면 → "액세스 허용" 클릭
> (다른 기기에서 대시보드 접속하려면 필요). 개인 네트워크만 체크해도 됨.

---

## 4. 자동 시작 (PC 켜면 봇 자동 실행) — 선택

**작업 스케줄러**:
1. `Win + R` → `taskschd.msc`
2. "기본 작업 만들기" → 이름: `ICT Bot`
3. 트리거: **컴퓨터 시작 시** (또는 로그온 시)
4. 동작: 프로그램 시작 → `windows-run.bat` 전체 경로 지정
5. 완료

또는 더 간단히: `windows-run.bat` 바로가기를 시작프로그램 폴더
(`Win+R` → `shell:startup`)에 넣기.

---

## 5. 운영 명령 / 업데이트

```powershell
:: 코드 업데이트 (GitHub 방식일 때)
cd TradingBot
git pull
:: 그 후 windows-run.bat 다시 실행

:: 설정 변경
notepad config\config.yaml
:: 저장 후 봇 러너 창 닫고 windows-run.bat 재실행
```

---

## 참고

- **포트 8080** 사용 (5000은 일부 환경서 충돌 가능). 바꾸려면 windows-run.bat 의 `--port` 수정.
- **진입 0건이 정상** — Kill Zone(런던 02–05, 뉴욕 07–10 UTC = 한국 11–14시·16–19시)에 70점+ 셋업이 나와야 진입.
- **자동학습**은 거래 20건 이상 쌓이면 작동 (logs\learned_params.yaml 에 기록).
- PC가 켜져 있는 동안만 돈다. 24시간 운영하려면 PC 절전모드 해제 권장
  (설정 → 시스템 → 전원 → 절전: 안 함).
- 페이퍼 상태(잔고/포지션/학습)는 `logs\` 폴더에 영속 저장 — PC 재시작해도 이어짐.
