"""1280x400 black-and-white banner: pixel blackletter name, mono tagline, two dithered public-domain engravings."""
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "art"
SRC = {  # Wikimedia Commons, public domain (credited in NOTICE.md)
    "dore.png": "https://upload.wikimedia.org/wikipedia/commons/a/af/Confusion_of_Tongues.png",
    "durer.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Albrecht_D%C3%BCrer%2C_Saint_Jerome_in_His_Study%2C_1514%2C_NGA_6642.jpg/960px-Albrecht_D%C3%BCrer%2C_Saint_Jerome_in_His_Study%2C_1514%2C_NGA_6642.jpg",
    "pirata.ttf": "https://github.com/google/fonts/raw/main/ofl/pirataone/PirataOne-Regular.ttf",
}
W, H, INK, ART = 1280, 400, 232, 120


def get(name):
    p = CACHE / name
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(SRC[name], headers={"User-Agent": "skilladmit-banner/0.1"})
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return p


def engraving(name, box, width, fade_left, gamma):
    """Crop, invert, scale to banner height, fade toward the centre, then 2-tone Floyd-Steinberg dither."""
    im = Image.open(get(name)).convert("L").crop(box)
    im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS).crop((0, 0, width, H))
    a = (1 - np.asarray(im, float) / 255) ** gamma  # inverted: engraved lines become ink on black
    lo, hi = np.percentile(a, (45, 99))
    a = np.clip((a - lo) / (hi - lo), 0, 1)  # keep only the darkest engraved strokes
    ramp = np.linspace(1, 0, width) if fade_left else np.linspace(0, 1, width)
    a = a * np.clip(ramp * 1.6, 0, 1)[None, :] ** 1.3
    bw = Image.fromarray((a * 255).astype(np.uint8)).convert("1")  # PIL's default "1" conversion dithers (FS)
    return Image.fromarray(np.where(np.asarray(bw), ART, 0).astype(np.uint8))


def pixel_text(text, font, size, scale):
    f = ImageFont.truetype(str(font), size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + 2, b - t + 2), 0)
    ImageDraw.Draw(im).text((1 - l, 1 - t), text, font=f, fill=255)
    im = im.point(lambda v: INK if v > 110 else 0)
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def main():
    out = Image.new("L", (W, H), 0)
    out.paste(engraving("dore.png", (40, 40, 475, 560), 380, True, 1.1), (0, 0))
    out.paste(engraving("durer.jpg", (360, 380, 880, 940), 380, False, 1.1), (W - 380, 0))
    name = pixel_text("skilladmit", get("pirata.ttf"), 46, 3)
    out.paste(name, ((W - name.width) // 2, 92))
    mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    for i, line in enumerate(("a calibrated gate for skills that", "would steal the routes that already work")):
        t = pixel_text(line, mono, 11, 2)
        out.paste(t.point(lambda v: 170 if v else 0), ((W - t.width) // 2, 250 + i * 34))
    dst = ROOT / "docs" / "screenshots" / "banner.png"
    out.convert("RGB").save(dst)
    print(dst, out.size)


if __name__ == "__main__":
    main()
