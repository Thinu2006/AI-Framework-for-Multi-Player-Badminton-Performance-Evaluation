import os
import cv2
import sys
import time
import numpy as np
import tensorflow as tf
from collections import deque

try:
    from .parser import parser
    from .TrackNet import ResNet_Track
except ImportError:
    from parser import parser
    from TrackNet import ResNet_Track

# Parse arguments (defaults)
args = parser.parse_args()
HEIGHT = args.HEIGHT
WIDTH = args.WIDTH
BATCH_SIZE = 1
FRAME_STACK = args.frame_stack
load_weights = args.load_weights
video_path = args.video_path

# Model setup
model = None

def get_prediction_model():
    global model
    if model is None:
        model = ResNet_Track(input_shape=(3, HEIGHT, WIDTH))
        if load_weights:
            try:
                model.load_weights(load_weights).expect_partial()
                print("Load weights successfully")
            except Exception:
                print("Fail to load weights, please modify path in parser.py --load_weights")
    return model

def run_shuttle_prediction():
    """Run the shuttle tracking prediction pipeline."""
    if not os.path.isfile(video_path) or not video_path.endswith('.mp4'):
        print("Not a valid video path! Please modify path in parser.py --video_path")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Initialize buffers
    gray_imgs = deque()
    success, image = cap.read()
    if not success:
        print("Failed to read first frame")
        return
    ratio = image.shape[0] / HEIGHT

    size = (int(WIDTH * ratio), int(HEIGHT * ratio))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{video_name}_predict.mp4", fourcc, fps, size)
    out.write(image)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=2)
    gray_imgs.append(img)

    for _ in range(FRAME_STACK - 1):
        success, image = cap.read()
        if not success:
            break
        out.write(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        gray_imgs.append(img)

    frame_no = FRAME_STACK - 1
    while success:
        if frame_no >= n_frames:
            break
        img_input = np.concatenate(gray_imgs, axis=2)
        img_input = cv2.resize(img_input, (WIDTH, HEIGHT))
        img_input = np.moveaxis(img_input, -1, 0)
        img_input = np.expand_dims(img_input, axis=0).astype('float') / 255.

        y_pred = get_prediction_model().predict(img_input, batch_size=BATCH_SIZE)
        y_pred = (y_pred > 0.5).astype('float32')

        h_pred = (y_pred[0] * 255).astype('uint8')
        if np.amax(h_pred) <= 0:
            out.write(image)
        else:
            cnts, _ = cv2.findContours(h_pred[0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for i in range(1, len(rects)):
                area = rects[i][2] * rects[i][3]
                if area > max_area:
                    max_area_idx = i
                    max_area = area
            target = rects[max_area_idx]
            cx_pred = int(ratio * (target[0] + target[2] / 2))
            cy_pred = int(ratio * (target[1] + target[3] / 2))
            image_cp = np.copy(image)
            cv2.circle(image_cp, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
            out.write(image_cp)

        success, image = cap.read()
        if success:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)
            gray_imgs.append(img)
            gray_imgs.popleft()
            frame_no += 1

    out.release()

if __name__ == "__main__":
    run_shuttle_prediction()
