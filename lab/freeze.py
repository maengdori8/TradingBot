"""데이터 스냅샷 동결 + 해시. 이후 모든 분석은 동결본만 사용한다."""
from __future__ import annotations
import hashlib, json
import pandas as pd

OUT = 'lab/frozen'
import os; os.makedirs(OUT, exist_ok=True)

def clean_funding():
    f = pd.read_parquet('lab/data/broad_funding_8h.parquet')
    f = f[f.index >= pd.Timestamp('2019-01-01', tz='utc')]     # 손상 타임스탬프 제거
    return f.sort_index()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

if __name__ == '__main__':
    man = {}
    f = clean_funding(); f.to_parquet(f'{OUT}/funding.parquet')
    for src, dst in [('lab/data/broad_perp_1d.parquet', 'perp_1d.parquet'),
                     ('lab/data/broad_spot_1d.parquet', 'spot_1d.parquet'),
                     ('lab/data/perp_1h.parquet', 'perp_1h.parquet'),
                     ('lab/data/spot_1h.parquet', 'spot_1h.parquet'),
                     ('lab/data/universe_linear.parquet', 'universe.parquet')]:
        pd.read_parquet(src).to_parquet(f'{OUT}/{dst}')
    for p in sorted(os.listdir(OUT)):
        if p.endswith('.parquet'): man[p] = sha(f'{OUT}/{p}')
    man['_frozen_at'] = pd.Timestamp.utcnow().isoformat()
    man['_funding_shape'] = list(f.shape)
    json.dump(man, open(f'{OUT}/MANIFEST.json', 'w'), indent=2)
    print(json.dumps(man, indent=2))
