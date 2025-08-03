"""
coord_recorder.py – quick-and-simple OSRS coordinate recorder
──────────────────────────────────────────────────────────────
•  Hotkeys:
     F9   → record a coordinate
     ESC  → quit recorder
•  Also provides a   “Record”   button in the tiny Tk window.
Captured points are appended to a JSON file:
      <PROJECT_ROOT>/assets/coordinates/coords.json
"""

from pathlib import Path
import json
import tkinter as tk
from threading import Thread

import pyautogui
from pynput import keyboard

# ─────────────────────────── settings ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent       # …/osrs
COORD_DIR    = PROJECT_ROOT / "assets" / "coordinates"
COORD_FILE   = COORD_DIR / "coords.json"
HOTKEY_KEY   = keyboard.Key.f9                       # key to save

# --------------------------------------------------------------- #

# Make sure target folder exists
COORD_DIR.mkdir(parents=True, exist_ok=True)

# Load any existing list
try:
    coords = json.loads(COORD_FILE.read_text())
except (FileNotFoundError, json.JSONDecodeError):
    coords = []  # plain list of [x, y]

# --------------- helper to append & persist -------------------- #
def save_point(pt):
    coords.append(list(pt))
    COORD_FILE.write_text(json.dumps(coords, indent=2))
    print(f"[+] recorded: {pt}  (total {len(coords)})")

# --------------- hotkey listener thread ------------------------ #
def hotkey_listener():
    def on_press(key):
        if key == HOTKEY_KEY:
            save_point(pyautogui.position())
        elif key == keyboard.Key.esc:
            print("[*] ESC pressed – exiting recorder.")
            root.quit()          # close Tk
            return False         # stop listener

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

Thread(target=hotkey_listener, daemon=True).start()

# --------------- tiny Tk window (status + record btn) ----------- #
root      = tk.Tk()
root.title("OSRS Coord Recorder")
root.attributes("-topmost", True)
root.resizable(False, False)

status_var = tk.StringVar(value=f"{len(coords)} pts saved")

tk.Label(root, text="Press F9 or click:", font=("Consolas", 11)).pack(padx=8, pady=(8, 2))
tk.Button(root, text="Record", width=12,
          command=lambda: (save_point(pyautogui.position()),
                           status_var.set(f"{len(coords)} pts saved"))
          ).pack(pady=2)
tk.Label(root, textvariable=status_var).pack(pady=(2, 6))
tk.Label(root, text="ESC → quit", fg="grey").pack(pady=(0, 4))

# update status every half-second so F9 counts appear
def refresh():
    status_var.set(f"{len(coords)} pts saved")
    root.after(500, refresh)
refresh()

root.mainloop()