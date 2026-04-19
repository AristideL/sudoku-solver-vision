import os
import time
from datetime import datetime
import mss
import mss.tools

# dossier principal
BASE_FOLDER = "screenshots"

# dossier avec timestamp
session_name = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
session_folder = os.path.join(BASE_FOLDER, session_name)

os.makedirs(session_folder, exist_ok=True)

print(f"Saving screenshots in: {session_folder}")

with mss.mss() as sct:
    monitor = sct.monitors[1]  # écran principal

    i = 0
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"screen_{i:05d}_{timestamp}.png"
        filepath = os.path.join(session_folder, filename)

        screenshot = sct.grab(monitor)
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=filepath)

        print(f"Saved: {filepath}")

        i += 1
        time.sleep(0.5)