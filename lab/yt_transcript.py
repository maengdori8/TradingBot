"""유튜브 자막 추출기 — innertube player API + timedtext v3 파싱."""
from __future__ import annotations
import json, re, html, sys, time, urllib.request

def _open(req, tries=5):
    for i in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=30)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))

def player(vid):
    body = {"context": {"client": {"clientName": "ANDROID", "clientVersion": "20.10.38"}}, "videoId": vid}
    req = urllib.request.Request("https://www.youtube.com/youtubei/v1/player",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "com.google.android.youtube/20.10.38"})
    with _open(req) as r:
        return json.loads(r.read())

def transcript(vid):
    d = player(vid)
    det = d.get("videoDetails", {})
    tracks = d.get("captions", {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
    if not tracks:
        return det.get("title", "?"), int(det.get("lengthSeconds", 0)), None
    with _open(tracks[0]["baseUrl"]) as r:
        raw = r.read().decode("utf-8", errors="ignore")
    # timedtext v3: <p ...>텍스트 or <s>단어</s>들</p>
    ps = re.findall(r"<p[^>]*>(.*?)</p>", raw, re.S)
    out = []
    for p in ps:
        p = re.sub(r"<s[^>]*>", "", p).replace("</s>", "")
        p = html.unescape(p).replace("\n", " ")
        if p.strip():
            out.append(p.strip())
    text = " ".join(" ".join(out).split())
    return det.get("title", "?"), int(det.get("lengthSeconds", 0)), text

if __name__ == "__main__":
    for vid in sys.argv[1:]:
        t, sec, txt = transcript(vid)
        n = len(txt) if txt else 0
        print(f"{vid} | {sec//60}분 | {n}자 | {t}")
        if txt:
            open(f"/tmp/yt_{vid}.txt", "w").write(txt)
        time.sleep(1)
