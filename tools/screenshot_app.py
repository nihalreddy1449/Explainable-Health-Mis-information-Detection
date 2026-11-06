"""Full-page screenshots of the Streamlit demo, via headless Chrome + DevTools protocol.

Chrome's plain --screenshot flag fires before Streamlit finishes rendering over its
websocket, so this drives Chrome directly and waits for a real DOM condition (a
verdict, or the example selector for the empty page) with no spinner still showing.

Usage:
    python tools/screenshot_app.py out.png "http://localhost:8501/?claim=..."  [width] [wait_for]
"""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9333


class CDP:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        mid = self.n
        await self.ws.write_message(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True)
        return r.get("result", {}).get("value")


async def shoot(out: Path, url: str, width: int, wait_for: str, clip_from=None, clip_to=None,
                timeout: float = 240):
    ver = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version").read())
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?about:blank", method="PUT")
    target = json.loads(urllib.request.urlopen(req).read())
    ws = await websocket_connect(target["webSocketDebuggerUrl"], max_message_size=200 * 1024 * 1024)
    cdp = CDP(ws)
    await cdp.call("Page.enable")
    await cdp.call("Emulation.setDeviceMetricsOverride", width=width, height=1000,
                   deviceScaleFactor=2, mobile=False)
    await cdp.call("Page.navigate", url=url)

    # Streamlit's spinner test-id differs across versions, so also check the visible
    # spinner text. The page is only "done" when the target exists and nothing is busy.
    cond = (f"(() => {{ const ok = document.querySelector({json.dumps(wait_for)});"
            f" const t = document.body.innerText;"
            f" const busy = !!document.querySelector('[data-testid=\"stSpinner\"], .stSpinner')"
            f"   || t.includes('Computing word attributions') || t.includes('Verifying...')"
            f"   || t.includes('Loading the verifier') || t.includes('Assessing the article')"
            f"   || t.includes('Loading the models');"
            f" return !!ok && !busy; }})()")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if await cdp.js(cond):
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(f"'{wait_for}' never appeared at {url}")
    await asyncio.sleep(1.2)  # let fonts and bar widths settle

    # Streamlit scrolls an inner container, so document height is just the viewport.
    # Take the tallest scrollHeight of any element instead.
    h = await cdp.js("Math.ceil(Math.max(...[...document.querySelectorAll('*')]"
                     ".map(e => e.scrollHeight)))")
    await cdp.call("Emulation.setDeviceMetricsOverride", width=width, height=int(h) + 40,
                   deviceScaleFactor=2, mobile=False)
    await asyncio.sleep(0.8)
    params = {"format": "png", "captureBeyondViewport": True}
    if clip_from:
        box = await cdp.js(
            f"(() => {{ const a = document.querySelector({json.dumps(clip_from)});"
            f" const b = document.querySelector({json.dumps(clip_to or clip_from)});"
            f" if (!a || !b) return null;"
            f" const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();"
            f" const col = document.querySelector('[data-testid=\"stMainBlockContainer\"]')"
            f"   .getBoundingClientRect();"
            f" return [col.left, ra.top, col.width, rb.bottom - ra.top]; }})()")
        if not box:
            raise RuntimeError(f"clip selectors not found: {clip_from} / {clip_to}")
        x, y, w, hh = box
        params["clip"] = {"x": max(x - 24, 0), "y": max(y - 18, 0), "width": w + 48,
                          "height": hh + 36, "scale": 1}
    shot = await cdp.call("Page.captureScreenshot", **params)
    out.write_bytes(base64.b64decode(shot["data"]))
    ws.close()
    print(f"saved {out}  ({width}px wide, {h}px tall)")


def main():
    out, url = Path(sys.argv[1]), sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 1100
    wait_for = sys.argv[4] if len(sys.argv) > 4 else ".foot"
    clip_from = sys.argv[5] if len(sys.argv) > 5 else None
    clip_to = sys.argv[6] if len(sys.argv) > 6 else None
    prof = tempfile.mkdtemp(prefix="cdp-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={prof}", "--hide-scrollbars", "--no-first-run",
                             "--disable-gpu", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        asyncio.run(shoot(out, url, width, wait_for, clip_from, clip_to))
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
