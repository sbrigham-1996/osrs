from pathlib import Path
import cv2
import numpy as np
import pyautogui
import time
import random
import os
import tkinter as tk
import threading

pyautogui.FAILSAFE = False

class Overlay(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OSRS Bot")
        self.attributes('-topmost', True)
        self.geometry('220x140+20+40')
        self.resizable(False, False)
        self.configure(bg='#111')
        self.start_time = time.time()
        self.total_logs = 0
        self.current_logs = 0
        self.lbl_state = tk.Label(self, text='Idle', fg='#7cf', bg='#111', font=('Helvetica', 12))
        self.lbl_state.pack(pady=(6, 1))

        self.lbl_runtime = tk.Label(self, text='Run time: 00:00:00',
                                    fg='#ccc', bg='#111', font=('Helvetica', 10))
        self.lbl_runtime.pack()
        self.lbl_inv = tk.Label(self, text='Logs: 0/27', fg='#ccc', bg='#111', font=('Helvetica', 10))
        self.lbl_inv.pack()
        self.lbl_total = tk.Label(self, text='Total Logs: 0', fg='#ccc',
                                  bg='#111', font=('Helvetica', 10))
        self.lbl_exp = tk.Label(self, text='XP/hr: 0', fg='#ccc',
                                bg='#111', font=('Helvetica', 10))
        self.lbl_exp.pack()
    def set_state(self, txt: str):
        self.lbl_state.config(text=txt)
        self.update_idletasks()
    def update_runtime(self):
        """Update the runtime label in HH:MM:SS format."""
        elapsed = time.time() -self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        self.lbl_runtime.config(text=f'Run Time: {hours:02d}:{minutes:02d}:{seconds:02d}')
    def update_stats(self):
        """Refresh runtime, total logs and XP/hr based on current counters."""
        self.update_runtime()
        self.lbl_total.config(text=f'Total Logs: {self.total_logs}')
        elapsed_hours = max((time.time() - self.start_time) / 3600.0, 1e-6)
        xp = (self.total_logs + self.current_logs) * 100.0
        xp_per_hour = xp / elapsed_hours
        self.lbl_exp.config(text=f'XP/hr: {int(xp_per_hour)}')
        self.update_idletasks()

    def set_inv(self, n: int, total: int = 27):
        """
        Update the current inventory count. This will also refresh related
        statistics such as runtime and XP/hour.
        """
        self.current_logs = n
        self.lbl_inv.config(text=f'Logs: {n}/{total}')
        self.update_stats()

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

MAPLES_MARKER      = ASSETS_DIR / "minimap" / "MAPLE_ICON_PATH.png"
BANK_MARKER        = ASSETS_DIR / "minimap" / "BANK_MINIMAP_MARKER.png"
BANKER_ICON_PATH   = ASSETS_DIR / "actions" / "BANKER_ICON_PATH.png"
BANKER_ICON_PATH2   = ASSETS_DIR / "actions" / "BANKER_ICON_PATH2.png"
DEPOSIT_ICON_PATH  = ASSETS_DIR / "actions" / "DEPOSIT_ICON_PATH.png"

# Fletching tool
KNIFE_ICON_PATH = ASSETS_DIR / "actions" / "knife.png"

# Reset ground spot for click reset
RESET_GROUND_PATH = ASSETS_DIR / "ground" / "reset.png"


# Tree templates (tree1..tree5)
TREE_TEMPLATES = [ASSETS_DIR / "trees" / f"tree{i}.png" for i in range(1, 4)]

# Inventory detection region (top-left x,y + width,height)
# You gave: top-left=(996,260), top-right=(1171,513) -> we use (996,260,175,253)
INV_REGION = (840, 260, 180, 255)  # (x, y, w, h) – previously working crop

# Candidate log icons (first one found will be used)
LOG_ICON_CANDIDATES = [
    ASSETS_DIR / "actions" / "maple_log.png",
    ASSETS_DIR / "actions" / "log_icon.png",
    ASSETS_DIR / "actions" / "yew_log.png",
]

# Optional: limit minimap searches to a small area

MINIMAP_REGION = None  # e.g., (screen_x, screen_y, width, height)
TREES_REGION = None  # e.g., (x, y, w, h) to limit tree searches

BANK_RUNBACK_WAIT = 13  # seconds to wait after clicking bank minimap marker before attempting banker


# ---------- Generic image helpers ----------
def locate_image(path, confidence=0.8, region=None):
    """Return (x,y) center or None."""
    try:
        return pyautogui.locateCenterOnScreen(str(path), confidence=confidence, region=region)
    except Exception:
        return None

def click_image(path, confidence=0.6, region=None, jitter=1, move_dur=(0.15, 0.25)):
    """Locate and click an image with small jitter. Return True if clicked."""
    pt = locate_image(path, confidence=confidence, region=region)
    if not pt:
        return False
    x, y = pt
    x += random.randint(-jitter, jitter)
    y += random.randint(-jitter, jitter)
    pyautogui.moveTo(x, y, duration=random.uniform(*move_dur))
    pyautogui.click()
    return True

def click_point(pt, jitter=1, move_dur=(0.15, 0.25)):
    """Move to a point with small jitter and click."""
    x, y = pt
    x += random.randint(-jitter, jitter)
    y += random.randint(-jitter, jitter)
    pyautogui.moveTo(x, y, duration=random.uniform(*move_dur))
    pyautogui.click()



def click_minimap_marker(marker_path, label: str = "MARKER", tries: int = 6,
                         confidence_sequence=(0.70, 0.60, 0.55), region=MINIMAP_REGION):
    """Try several times/thresholds to click a minimap marker. Return True if clicked."""
    for _ in range(tries):
        for conf in confidence_sequence:
            pt = locate_image(marker_path, confidence=conf, region=region)
            if pt:
                x, y = pt
                # No jitter for precise minimap clicks
                pyautogui.moveTo(x, y, duration=random.uniform(0.22, 0.35))
                pyautogui.click()
                print(f"→ Clicked {label} minimap marker (conf≈{conf:.2f}).")
                return True
        time.sleep(0.35)


# ------------------- Additional helpers -------------------
def wait_for_image(path, timeout=10, confidence=0.8, region=None, poll=0.5):
    """Return True as soon as image is seen within timeout; else False."""
    end = time.time() + timeout
    while time.time() < end:
        if locate_image(path, confidence=confidence, region=region):
            return True
        time.sleep(poll)
    return False

def click_until(path, tries=5, confidence=0.6, region=None, interval=1.0, jitter=1):
    """Try clicking an image up to N times; return True on first success."""
    for _ in range(tries):
        pt = locate_image(path, confidence=confidence, region=region)
        if pt:
            x, y = pt
            x += random.randint(-jitter, jitter)
            y += random.randint(-jitter, jitter)
            pyautogui.moveTo(x, y, duration=random.uniform(0.15, 0.25))
            pyautogui.click()
            return True
        time.sleep(interval)
    return False

def any_tree_visible(confidence=0.52):
    """Fast check: is any tree template currently on screen?"""
    for t in TREE_TEMPLATES:
        if locate_image(t, confidence=confidence):
            return True
    return False

# ---------- Inventory detection (per-slot, robust) ----------
def _load_first_existing(paths):
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img, p
    return None, None

# --- Multi-scale template matching helper ---
def _match_max_multiscale(tmpl, roi, scales=(0.85, 0.9, 1.0, 1.1, 1.15)):
    """Return the best template-match score across a few scales for the given ROI."""
    best = -1.0
    for s in scales:
        # Resize ROI instead of template for speed
        if s != 1.0:
            resized = cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        else:
            resized = roi
        if resized.shape[0] < tmpl.shape[0] or resized.shape[1] < tmpl.shape[1]:
            continue
        res = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best:
            best = max_val
    return best

# --- Multi-scale, multi-template tree matching ---
def locate_any_tree_multiscale(region: tuple | None = TREES_REGION,
                               threshold: float = 0.70,
                               scales=(0.6, 0.8, 1.0, 1.2),
                               grayscale: bool = True):
    """Return (x, y) of the best-matching tree across templates/scales, or None."""
    # Grab screenshot once (optionally limited to region)
    if region is None:
        shot = pyautogui.screenshot()
        off_x, off_y = 0, 0
    else:
        shot = pyautogui.screenshot(region=region)
        off_x, off_y = region[0], region[1]
    shot_bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    shot_proc = cv2.cvtColor(shot_bgr, cv2.COLOR_BGR2GRAY) if grayscale else shot_bgr

    best_val = -1.0
    best_loc = None
    best_scale = 1.0
    best_size = (0, 0)  # (tw, th)

    for tpath in TREE_TEMPLATES:
        templ = cv2.imread(str(tpath), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if templ is None:
            continue
        th, tw = templ.shape[:2]
        for s in scales:
            resized = cv2.resize(shot_proc, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            if resized.shape[0] < th or resized.shape[1] < tw:
                continue
            res = cv2.matchTemplate(resized, templ, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = s
                best_size = (tw, th)

    if best_val >= threshold and best_loc is not None:
        tw, th = best_size
        x = int(best_loc[0] / best_scale + tw / (2 * best_scale)) + off_x
        y = int(best_loc[1] / best_scale + th / (2 * best_scale)) + off_y
        return (x, y)
    return None

def detect_log_count(
    region=INV_REGION,
    per_slot_conf: float = 0.58,
    pad: int = 14,
    scales=(0.85, 0.9, 1.0, 1.1, 1.15)
) -> int:
    """
    Count how many inventory slots (4x7) contain the log icon by matching inside
    a small window around each slot center. Uses multi‑scale matching per slot.
    Set env DEBUG_INV=1 to save a debug PNG of the inventory crop with scores.
    """
    tmpl, used_path = _load_first_existing(LOG_ICON_CANDIDATES)
    if tmpl is None:
        print("[err] No log-icon template found:", [str(p) for p in LOG_ICON_CANDIDATES])
        return 0

    # Capture just the inventory crop
    shot = pyautogui.screenshot(region=region)
    gray = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2GRAY)

    t_h, t_w = tmpl.shape[:2]
    # region is (x, y, w, h); we only need w,h for slot layout
    _, _, w, h = region
    cols, rows = 4, 7
    cell_w = w / cols
    cell_h = h / rows

    count = 0
    debug = os.environ.get("DEBUG_INV", "0") == "1"
    dbg_img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR) if debug else None

    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)

            # Window around slot center
            x1 = max(0, cx - (t_w // 2 + pad))
            y1 = max(0, cy - (t_h // 2 + pad))
            x2 = min(gray.shape[1], cx + (t_w // 2 + pad))
            y2 = min(gray.shape[0], cy + (t_h // 2 + pad))
            roi = gray[y1:y2, x1:x2]
            if roi.shape[0] < t_h or roi.shape[1] < t_w:
                continue

            score = _match_max_multiscale(tmpl, roi, scales=scales)

            if debug and dbg_img is not None:
                color = (0, 255, 0) if score >= per_slot_conf else (0, 0, 255)
                cv2.rectangle(dbg_img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(dbg_img, f"{score:.2f}", (x1, max(10, y1 - 2)),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)

            if score >= per_slot_conf:
                count += 1

    if debug and dbg_img is not None:
        dbg_dir = ASSETS_DIR / "debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        out_path = dbg_dir / f"inv_debug_{int(time.time())}.png"
        cv2.imwrite(str(out_path), dbg_img)
        print(f"[debug] saved {out_path}")

    return count

def inventory_full(threshold: int = 27) -> bool:
    """True if at least `threshold` slots look like logs."""
    return detect_log_count() >= threshold

def inventory_empty() -> bool:
    """True if no slots look like logs."""
    return detect_log_count() == 0

# ---------- New bulletproof structure ----------

def random_tree_not_last(last_idx, n=4):
    idxs = list(range(n))
    if last_idx in idxs:
        idxs.remove(last_idx)
    return random.choice(idxs)

def run_to_maples(overlay=None):
    print("→ Clicking minimap to run to maples…")
    if overlay:
        overlay.set_state('Running to maples…')
    click_minimap_marker(MAPLES_MARKER, label="MAPLES", tries=8, confidence_sequence=(0.72, 0.62, 0.54))
    time.sleep(random.uniform(10.5, 13.5))

def chop_trees_until_full(overlay=None):
    print("→ Randomly chopping trees until full inventory…")
    last_tree = -1
    while not inventory_full():
        idx = random_tree_not_last(last_tree, n=4)
        last_tree = idx
        if idx >= len(TREE_TEMPLATES):
            print(f"[warn] Tree template idx {idx} not available.")
            continue
        tree_path = TREE_TEMPLATES[idx]
        print(f"→ Clicking tree {idx+1} ({tree_path.name})")
        pt = locate_image(tree_path, confidence=0.6)
        if pt:
            click_point(pt)
        else:
            print("→ Tree not found, searching another...")
            continue
        if overlay:
            overlay.set_state(f'Chopping tree {idx+1}')
        # Chop for 19 seconds at this tree
        for _ in range(19):
            if overlay:
                # Update inventory count and stats (detect_log_count is used
                # even though it is relatively expensive; update once per second).
                overlay.set_inv(detect_log_count())
            if inventory_full():
                break
            time.sleep(1)
    print("→ Inventory full after chopping.")
    # After loop exit, update overlay with full inventory count (should be 28).
    if overlay:
        overlay.set_inv(detect_log_count())

def click_reset_spot():
    print("→ Clicking reset spot…")
    click_image(RESET_GROUND_PATH, confidence=0.7)
    time.sleep(random.uniform(2.0, 3.0))

def run_to_bank(overlay=None):
    print("→ Running to bank via minimap…")
    if overlay:
        overlay.set_state('Running to bank…')
    click_minimap_marker(BANK_MARKER, label="BANK", tries=8, confidence_sequence=(0.72, 0.62, 0.54))
    time.sleep(random.uniform(10.5, 13.5))

def bank_and_deposit(overlay=None):
    print("→ At bank, opening bank UI…")
    if overlay:
        overlay.set_state('Banking…')
    time.sleep(2)
    for i in range(3):
        if click_image(BANKER_ICON_PATH, confidence=0.72):
            print("→ Banker clicked.")
            if wait_for_image(DEPOSIT_ICON_PATH, timeout=7, confidence=0.7):
                break
        time.sleep(0.25)
    # Close bank UI to fletch logs (knife already in inventory)
    pyautogui.press('esc')
    time.sleep(0.5)
    # Fletch all logs in inventory
    print("→ Fletching logs…")
    log_count = detect_log_count()
    for _ in range(log_count):
        # Select knife in inventory
        knife_pt = locate_image(KNIFE_ICON_PATH, confidence=0.7, region=INV_REGION)
        if knife_pt:
            click_point(knife_pt)
            time.sleep(0.1)
        # Click on a log to fletch
        log_pt = locate_image(LOG_ICON_CANDIDATES[0], confidence=0.7, region=INV_REGION)
        if log_pt:
            click_point(log_pt)
            time.sleep(0.1)
    time.sleep(1)
    # Re-open bank to deposit fletched items
    print("→ Re-opening bank to deposit…")
    click_minimap_marker(BANK_MARKER, label="BANK", tries=8, confidence_sequence=(0.72, 0.62, 0.54))
    time.sleep(random.uniform(10.5, 13.5))
    print("→ Clicking deposit all…")
    logs_before = detect_log_count()
    for i in range(3):
        if click_image(DEPOSIT_ICON_PATH, confidence=0.7):
            time.sleep(0.25)
            if inventory_empty():
                break
        time.sleep(0.7)
    if inventory_empty():
        print("→ Deposit complete.")
        if overlay:
            overlay.total_logs += logs_before
            overlay.set_inv(0)
            overlay.set_state('Deposited')
    else:
        print("[WARN] Inventory not empty after deposit — check template/confidence.")

def main_loop(overlay):
    # Start at bank, then always go to maples first
    while True:
        run_to_maples(overlay=overlay)
        # Add a random "arrival settle" pause to simulate human reaction time at maples
        settle_time = random.uniform(1.7, 3.7)
        print(f"→ Waiting {settle_time:.1f}s for arrival/settle at maples…")
        time.sleep(settle_time)
        time.sleep(1)
        chop_trees_until_full(overlay=overlay)
        click_reset_spot(overlay=overlay)
        # Fletch logs here before running to bank (knife in slot #1)
        overlay.set_state('Fletching logs post-reset…')
        print("→ Fletching logs post-reset…")
        log_count = detect_log_count()
        for _ in range(log_count):
            knife_pt = locate_image(KNIFE_ICON_PATH, confidence=0.7, region=INV_REGION)
            if knife_pt:
                click_point(knife_pt)
                time.sleep(0.1)
            log_pt = locate_image(LOG_ICON_CANDIDATES[0], confidence=0.7, region=INV_REGION)
            if log_pt:
                click_point(log_pt)
                time.sleep(0.1)
        time.sleep(1)
        run_to_bank(overlay=overlay)
        bank_and_deposit(overlay=overlay)

def main():
    print("Starting OSRS Maple-chop + Bank loop with overlay…")
    overlay = Overlay()
    t = threading.Thread(target=main_loop, args=(overlay,), daemon=True)
    t.start()
    overlay.mainloop()

if __name__ == "__main__":
    main()