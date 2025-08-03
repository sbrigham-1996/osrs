# calibrate_inventory.py
from pathlib import Path
import json
import time
import pyautogui
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
OUT_JSON = BASE_DIR / "inventory_region.json"
OUT_PREVIEW = BASE_DIR / "debug_inventory_overlay.png"

def capture_point(prompt):
    input(f"\n{prompt}\nMove the mouse to the spot, then press ENTER here…")
    x, y = pyautogui.position()
    print(f"  captured: ({x}, {y})")
    return x, y

def draw_grid(img, cols=4, rows=7, color=(0,255,0), width=2):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    # rectangle border
    draw.rectangle([(0,0), (w-1,h-1)], outline=color, width=width)
    # verticals
    for c in range(1, cols):
        x = int(c * w / cols)
        draw.line([(x,0),(x,h)], fill=color, width=width)
    # horizontals
    for r in range(1, rows):
        y = int(r * h / rows)
        draw.line([(0,y),(w,y)], fill=color, width=width)

def main():
    print("Inventory region calibration")
    print("Tip: Open the inventory so the 4×7 slots are fully visible.\n")

    tlx, tly = capture_point("1) TOP-LEFT corner of the inventory (outside edge of the slot grid).")
    brx, bry = capture_point("2) BOTTOM-RIGHT corner of the inventory (outside edge of the slot grid).")

    # Normalize (user might click in any order)
    x1, y1 = min(tlx, brx), min(tly, bry)
    x2, y2 = max(tlx, brx), max(tly, bry)
    w, h = x2 - x1, y2 - y1

    region = (x1, y1, w, h)
    print(f"\nComputed INV_REGION = {region}")

    # Save JSON
    with OUT_JSON.open("w") as f:
        json.dump({"INV_REGION": region}, f)
    print(f"Saved {OUT_JSON}")

    # Preview with grid overlay
    shot = pyautogui.screenshot(region=region)
    draw_grid(shot)
    shot.save(OUT_PREVIEW)
    print(f"Saved preview with grid overlay to {OUT_PREVIEW}")
    print("\nOpen the preview and check that the grid lines sit over slot boundaries.")
    print("If misaligned, re-run this script and recapture the corners.")

if __name__ == "__main__":
    main()