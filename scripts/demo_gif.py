"""Hero screenshot, live-mode screenshot and a demo GIF from the running dashboard (real committed run).

Needs `python3 app/server.py 8766` running. Uses Playwright with an installed Chrome/Chromium
(set CHROME=/path/to/chrome, or CDP=http://host:port to drive an already-open browser).
"""
import io, json, os, urllib.request
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
URL = os.environ.get("URL", "http://127.0.0.1:8766")


def frame(page):
    return Image.open(io.BytesIO(page.screenshot())).convert("L")


def main():
    T = json.load(urllib.request.urlopen(URL + "/api/traces"))
    e = max(T, key=lambda e: e["arms"]["always"]["broken"] - e["arms"]["skilladmit"]["broken"])
    n = len(e["arms"]["skilladmit"]["steps"])
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp(os.environ["CDP"]) if os.environ.get("CDP") else \
            p.chromium.launch(executable_path=os.environ.get("CHROME"))
        page = b.new_page()
        page.set_viewport_size({"width": 1600, "height": 960})
        page.goto(f"{URL}/?s={e['id']}")
        page.wait_for_function("typeof T !== 'undefined' && T && T.length")
        page.wait_for_timeout(400)
        frame(page).save(OUT / "hero.png")
        frames = []
        for t in range(n + 1):
            page.evaluate(f"t={t};draw()")
            im = frame(page).resize((1200, 720), Image.LANCZOS)
            frames += [im] * (3 if e["arms"]["skilladmit"]["steps"][max(0, t - 1)]["kind"] != "real" else 2)
        frames += [frames[-1]] * 12
        frames[0].save(OUT / "demo.gif", save_all=True, append_images=frames[1:], duration=220, loop=0, optimize=True)
        page.goto(f"{URL}/?s={e['id']}&live")
        page.wait_for_function("typeof T !== 'undefined' && T && T.length")
        page.click("#go")
        page.wait_for_function("document.getElementById('lbars').children.length > 0", timeout=60000)
        page.wait_for_timeout(300)
        frame(page).save(OUT / "live.png")
        b.close()
    print("demo stream", e["id"], "always broke", e["arms"]["always"]["broken"], "skilladmit broke",
          e["arms"]["skilladmit"]["broken"])


if __name__ == "__main__":
    main()
