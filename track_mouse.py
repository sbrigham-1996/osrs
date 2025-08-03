import pyautogui
import time

print("Move your mouse to a corner of the inventory. Ctrl-C to stop.")
try:
    while True:
        x, y = pyautogui.position()
        print(f"\rX={x:4d}, Y={y:4d}", end="", flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone.")