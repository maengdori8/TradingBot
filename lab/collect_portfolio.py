"""코호트 전 지갑의 과거 손익 곡선 수집 — 지속성 역사적 검증용.

지갑당 portfolio 요청 1회. 재실행 시 이미 수집된 지갑은 건너뛴다(재개 가능).
출력: logs/trader_portfolio.jsonl.gz (지갑당 1줄: allTime/perpAllTime 곡선)
"""
from __future__ import annotations
import gzip, json, time, urllib.request
from pathlib import Path

COHORT = Path('logs/trader_cohort.json.gz')
OUT = Path('logs/trader_portfolio.jsonl.gz')

def fetch(addr: str) -> dict | None:
    req = urllib.request.Request('https://api.hyperliquid.xyz/info',
        data=json.dumps({"type": "portfolio", "user": addr}).encode(),
        headers={'Content-Type': 'application/json'})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 * (i + 1))
    return None

def main() -> None:
    with gzip.open(COHORT, 'rt') as f:
        wallets = [w['address'] for w in json.load(f)['wallets']]
    done = set()
    if OUT.exists():
        with gzip.open(OUT, 'rt') as f:
            for line in f:
                try: done.add(json.loads(line)['address'])
                except Exception: pass
    todo = [w for w in wallets if w not in done]
    print(f"코호트 {len(wallets)} | 완료 {len(done)} | 남음 {len(todo)}", flush=True)
    t0 = time.time()
    with gzip.open(OUT, 'at') as out:
        for i, addr in enumerate(todo):
            d = fetch(addr)
            if d is not None and isinstance(d, list):
                keep = {}
                for name, v in d:
                    if name in ('allTime', 'perpAllTime'):
                        keep[name] = {'pnl': v.get('pnlHistory', []),
                                      'acct': v.get('accountValueHistory', [])}
                out.write(json.dumps({'address': addr, **keep}) + '\n')
            if (i + 1) % 200 == 0:
                out.flush()
                rate = (i + 1) / (time.time() - t0)
                eta = (len(todo) - i - 1) / max(rate, 0.1) / 60
                print(f"[{i+1}/{len(todo)}] {rate:.1f}/s ETA {eta:.0f}분", flush=True)
            time.sleep(0.18)          # ~5.5 req/s — 공개 한도의 절반 이하
    print(f"DONE {len(todo)}건, {(time.time()-t0)/60:.0f}분", flush=True)

if __name__ == '__main__':
    main()
