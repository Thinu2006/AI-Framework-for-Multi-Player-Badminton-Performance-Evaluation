"""
pose_estimation.py

MoveNet SinglePose Thunder Pose Estimator using TensorFlow Lite.
Detects 17 COCO body keypoints for detected player ROIs.
"""

import cv2
import numpy as np
import tensorflow as tf

KEYPOINT_DICT = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Keypoint connections for skeleton rendering
KEYPOINT_EDGES = [
    (0, 1, (255, 200, 0)),
    (0, 2, (255, 200, 0)),
    (1, 3, (255, 200, 0)),
    (2, 4, (255, 200, 0)),
    (5, 6, (0, 255, 255)),   # Shoulders
    (5, 7, (0, 255, 255)),   # Left arm
    (7, 9, (0, 255, 255)),
    (6, 8, (255, 0, 255)),   # Right arm
    (8, 10, (255, 0, 255)),
    (5, 11, (0, 255, 0)),    # Left torso
    (6, 12, (0, 255, 0)),    # Right torso
    (11, 12, (0, 255, 0)),   # Hips
    (11, 13, (0, 165, 255)), # Left leg
    (13, 15, (0, 165, 255)),
    (12, 14, (255, 100, 0)), # Right leg
    (14, 16, (255, 100, 0)),
]


class MoveNetPoseEstimator:
    """
    MoveNet SinglePose Thunder TFLite model wrapper.
    Inputs: [1, 256, 256, 3] uint8 RGB image tensor.
    Outputs: [1, 1, 17, 3] float32 tensor representing (y_norm, x_norm, score).
    """

    def __init__(self, model_path: str = "movenet_thunder.tflite"):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]  # [1, 256, 256, 3]
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

    def estimate_player_pose(self, frame: np.ndarray, player_bbox: tuple, margin: float = 0.15) -> dict:
        """
        Estimates pose keypoints for a player ROI.

        Parameters
        ----------
        frame : np.ndarray
            The full video frame (BGR).
        player_bbox : tuple
            (x1, y1, x2, y2) bounding box of the player.
        margin : float
            Safety margin around bounding box.

        Returns
        -------
        keypoints : dict
            Dictionary mapping keypoint_idx (0..16) -> {
                "name": str,
                "pt": (gx, gy) tuple in global frame pixels,
                "score": float confidence score
            }
        """
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = player_bbox
        pw = px2 - px1
        ph = py2 - py1

        # Add margin to crop area
        roi_x1 = max(0, int(px1 - pw * margin))
        roi_y1 = max(0, int(py1 - ph * margin))
        roi_x2 = min(w, int(px2 + pw * margin))
        roi_y2 = min(h, int(py2 + ph * margin))

        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0 or roi_w < 10 or roi_h < 10:
            return {}

        # Prepare 256x256 RGB input tensor
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(rgb_roi, (self.input_width, self.input_height))
        input_tensor = np.expand_dims(input_img, axis=0).astype(np.uint8)

        # Run TFLite inference
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])  # [1, 1, 17, 3]

        keypoints = {}
        for idx in range(17):
            y_norm = output_data[0, 0, idx, 0]
            x_norm = output_data[0, 0, idx, 1]
            score = float(output_data[0, 0, idx, 2])

            # Map normalized coordinates back to full image pixels
            gx = float(roi_x1 + x_norm * roi_w)
            gy = float(roi_y1 + y_norm * roi_h)

            keypoints[idx] = {
                "name": KEYPOINT_DICT[idx],
                "pt": (gx, gy),
                "score": score,
            }

        return keypoints


def draw_pose_skeleton(frame: np.ndarray, keypoints: dict, min_conf: float = 0.20) -> np.ndarray:
    """
    Draws MoveNet skeleton lines and joint dots on the frame.
    """
    output = frame.copy()
    if not keypoints:
        return output

    # 1. Draw Skeleton Edge Lines
    for edge in KEYPOINT_EDGES:
        idx1, idx2, color = edge
        kp1 = keypoints.get(idx1)
        kp2 = keypoints.get(idx2)

        if kp1 and kp2 and kp1["score"] >= min_conf and kp2["score"] >= min_conf:
            pt1 = (int(round(kp1["pt"][0])), int(round(kp1["pt"][1])))
            pt2 = (int(round(kp2["pt"][0])), int(round(kp2["pt"][1])))
            cv2.line(output, pt1, pt2, color, 2, cv2.LINE_AA)

    # 2. Draw Keypoint Joint Dots
    for idx, kp in keypoints.items():
        if kp["score"] >= min_conf:
            pt = (int(round(kp["pt"][0])), int(round(kp["pt"][1])))
            cv2.circle(output, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

    return output
