import os
import time
from datetime import datetime

import cv2
import numpy as np
import mss
from inference_sdk import InferenceHTTPClient

# ----------------------------
# Roboflow client
# ----------------------------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ssxmwUNGeLSJiWUIw4GM"
)

MODEL_ID = "gamezonedetection/2"

# ----------------------------
# folders
# ----------------------------
BASE = "dataset_capture"
RAW = os.path.join(BASE, "raw")
CROP = os.path.join(BASE, "gamezone")

os.makedirs(RAW, exist_ok=True)
os.makedirs(CROP, exist_ok=True)

# ----------------------------
# screen capture
# ----------------------------
with mss.mss() as sct:

    monitor = sct.monitors[1]  # main screen
    i = 0

    while True:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_path = os.path.join(RAW, f"screen_{i:05d}_{timestamp}.png")

        # capture
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        cv2.imwrite(raw_path, img)

        print("Captured:", raw_path)

        # ----------------------------
        # inference roboflow
        # ----------------------------
        result = CLIENT.infer(raw_path, model_id=MODEL_ID)

        preds = result.get("predictions", [])

        if len(preds) > 0:

            # prendre la meilleure detection
            pred = max(preds, key=lambda x: x["confidence"])

            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            crop = img[y1:y2, x1:x2]

            crop_path = os.path.join(
                CROP, f"gamezone_{i:05d}_{timestamp}.png"
            )

            cv2.imwrite(crop_path, crop)

            print("Game zone saved:", crop_path)

        else:
            print("No game zone detected")

        i += 1
        time.sleep(1)