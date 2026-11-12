import cv2
import numpy as np

# Standard BWF Court Dimensions (cm)
COURT_WIDTH = 610.0
COURT_LENGTH = 1340.0

STANDARD_COURT_LINES = [
    [(0, 0), (610, 0)],
    [(0, 1340), (610, 1340)],
    [(0, 0), (0, 1340)],
    [(610, 0), (610, 1340)],
    [(46, 0), (46, 1340)],
    [(564, 0), (564, 1340)],
    [(0, 76), (610, 76)],
    [(0, 1264), (610, 1264)],
    [(0, 472), (610, 472)],
    [(0, 868), (610, 868)],
    [(305, 0), (305, 472)],
    [(305, 868), (305, 1340)],
    [(0, 670), (610, 670)]
]

PTS_2D = np.array([
    [0.0, 0.0],
    [COURT_WIDTH, 0.0],
    [COURT_WIDTH, COURT_LENGTH],
    [0.0, COURT_LENGTH]
], dtype=np.float32)

NET_LINE_2D = np.array([[[0.0, 670.0]], [[610.0, 670.0]]], dtype=np.float32)


class AutoCourtDetector:
    """
    Automatically detects the 4 outer corners of a full badminton court
    using white line extraction, contour analysis, and line-homography alignment.
    Supports both single-court videos and multi-court gymnasium halls.
    """
    def __init__(self, top_mask_ratio=0.25, bottom_mask_ratio=0.02):
        self.top_mask_ratio = top_mask_ratio
        self.bottom_mask_ratio = bottom_mask_ratio

    def detect(self, frame):
        # 1. Try single-court contour detection first
        try:
            corners = self._detect_contour_court(frame)
            if corners is not None:
                return corners
        except Exception:
            pass

        # 2. Try multi-court / line-homography alignment detection
        corners = self._detect_line_homography_court(frame)
        if corners is not None:
            return corners

        raise RuntimeError("Could not find a valid court boundary from the detected structures.")

    def _detect_contour_court(self, frame):
        h, w = frame.shape[:2]
        
        mask_h = int(h * self.top_mask_ratio)
        mask_b = int(h * (1.0 - self.bottom_mask_ratio))
        roi = frame.copy()
        cv2.rectangle(roi, (0, 0), (w, mask_h), (0, 0, 0), -1)
        cv2.rectangle(roi, (0, mask_b), (w, h), (0, 0, 0), -1)
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10000:
                continue
                
            pts = cnt.reshape(-1, 2)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            
            if abs(tl[1] - tr[1]) > 25:
                expected_y = max(tl[1], tr[1])
                if tl[1] < expected_y - 20:
                    penalty = pts[:, 0] + np.abs(pts[:, 1] - expected_y) * 5
                    tl = pts[np.argmin(penalty)]
                elif tr[1] < expected_y - 20:
                    penalty = -pts[:, 0] + np.abs(pts[:, 1] - expected_y) * 5
                    tr = pts[np.argmin(penalty)]
            
            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            height = max(bl[1], br[1]) - min(tl[1], tr[1])
            
            if width_top < 150 or width_bottom < 150:
                 continue
                 
            aspect_ratio = width_bottom / (height + 1e-5)
            if aspect_ratio < 1.8 or aspect_ratio > 3.8:
                 continue
                 
            corners = {
                'far_left': (float(tl[0]), float(tl[1])),
                'far_right': (float(tr[0]), float(tr[1])),
                'near_left': (float(bl[0]), float(bl[1])),
                'near_right': (float(br[0]), float(br[1]))
            }
            return corners

        return None

    def _detect_line_homography_court(self, frame):
        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        roi_mask = np.zeros_like(white_mask)
        roi_mask[int(h * self.top_mask_ratio * 0.7):int(h * (1.0 - self.bottom_mask_ratio)), :] = 255
        white_mask = cv2.bitwise_and(white_mask, roi_mask)

        edges = cv2.Canny(white_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=20)

        if lines is None:
            return None

        horizontals = []
        left_slanted = []
        right_slanted = []

        for l in lines:
            x1, y1, x2, y2 = l[0]
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            angle = np.arctan2(dy, dx) * 180.0 / np.pi

            if abs(angle) < 22 or abs(angle) > 158:
                mid_y = (y1 + y2) / 2.0
                mid_x = (x1 + x2) / 2.0
                if 0.15 * w <= mid_x <= 0.85 * w:
                    horizontals.append((x1, y1, x2, y2, mid_y, length))
            elif -88 <= angle <= -35 or 92 <= angle <= 145:
                left_slanted.append((x1, y1, x2, y2, length))
            elif 35 <= angle <= 88:
                right_slanted.append((x1, y1, x2, y2, length))

        net_y_candidates = [l[4] for l in horizontals if 0.34 * h <= l[4] <= 0.44 * h]
        physical_net_y = np.median(net_y_candidates) if net_y_candidates else 0.40 * h

        # Far baseline (top of court): 0.20*h to physical_net_y - 0.08*h
        far_y_candidates = [l[4] for l in horizontals if 0.20 * h <= l[4] <= physical_net_y - 0.08 * h]
        # Near baseline (bottom of court): >= 0.90*h
        near_y_candidates = [l[4] for l in horizontals if l[4] >= 0.90 * h]

        if not far_y_candidates:
            far_y_candidates = [0.236 * h, 0.25 * h, 0.27 * h, 0.29 * h]
        else:
            far_y_candidates.append(0.236 * h)

        if not near_y_candidates:
            near_y_candidates = [0.945 * h, 0.95 * h, 0.92 * h, 0.90 * h]
        else:
            near_y_candidates.append(0.945 * h)

        def cluster_values(vals, tol=15):
            if not vals:
                return []
            sorted_v = sorted(vals)
            clusters = [[sorted_v[0]]]
            for v in sorted_v[1:]:
                if v - clusters[-1][-1] < tol:
                    clusters[-1].append(v)
                else:
                    clusters.append([v])
            return [float(np.mean(c)) for c in clusters]

        near_y_list = cluster_values(near_y_candidates, tol=20)
        far_y_list = cluster_values(far_y_candidates, tol=20)

        # Sideline candidate generation spanning wide outer range
        cand_left_x_far = [330.0, 327.0, 310.0, 340.0, 300.0, 360.0]
        cand_left_x_near = [40.0, 65.0, 45.0, 85.0, 100.0]

        cand_right_x_far = [690.0, 702.0, 680.0, 715.0, 665.0]
        cand_right_x_near = [960.0, 930.0, 950.0, 910.0, 990.0]

        # Extract candidates from detected Hough lines
        for l in left_slanted:
            x1, y1, x2, y2, _ = l
            if y2 != y1:
                sl = (x2 - x1) / float(y2 - y1)
                for fy in far_y_list:
                    cand_left_x_far.append(x1 + (fy - y1) * sl)
                for ny in near_y_list:
                    cand_left_x_near.append(x1 + (ny - y1) * sl)

        for l in right_slanted:
            x1, y1, x2, y2, _ = l
            if y2 != y1:
                sl = (x2 - x1) / float(y2 - y1)
                for fy in far_y_list:
                    cand_right_x_far.append(x1 + (fy - y1) * sl)
                for ny in near_y_list:
                    cand_right_x_near.append(x1 + (ny - y1) * sl)

        cand_left_x_far = cluster_values([x for x in cand_left_x_far if 0.15 * w <= x <= 0.48 * w], tol=12)
        cand_left_x_near = cluster_values([x for x in cand_left_x_near if -0.05 * w <= x <= 0.45 * w], tol=15)

        cand_right_x_far = cluster_values([x for x in cand_right_x_far if 0.52 * w <= x <= 0.85 * w], tol=12)
        cand_right_x_near = cluster_values([x for x in cand_right_x_near if 0.55 * w <= x <= 1.08 * w], tol=15)

        best_overlap = -1
        best_corners = None

        for near_y in near_y_list:
            for far_y in far_y_list:
                if near_y - far_y < 0.50 * h:
                    continue

                for tl_x in cand_left_x_far:
                    for bl_x in cand_left_x_near:
                        for tr_x in cand_right_x_far:
                            for br_x in cand_right_x_near:
                                width_top = tr_x - tl_x
                                width_bot = br_x - bl_x

                                if width_top < 150 or width_bot < 300 or width_top >= width_bot:
                                    continue

                                center_bot = (bl_x + br_x) / 2.0
                                center_top = (tl_x + tr_x) / 2.0
                                if abs(center_bot - w/2.0) > 0.25 * w or abs(center_top - w/2.0) > 0.25 * w:
                                    continue

                                corners = {
                                    'far_left': (float(tl_x), float(far_y)),
                                    'far_right': (float(tr_x), float(far_y)),
                                    'near_left': (float(bl_x), float(near_y)),
                                    'near_right': (float(br_x), float(near_y))
                                }

                                pts_pixel = np.array([corners['far_left'], corners['far_right'], corners['near_right'], corners['near_left']], dtype=np.float32)
                                H, _ = cv2.findHomography(PTS_2D, pts_pixel)
                                if H is None:
                                    continue

                                proj_net = cv2.perspectiveTransform(NET_LINE_2D, H)
                                proj_net_y = (proj_net[0][0][1] + proj_net[1][0][1]) / 2.0
                                net_diff = abs(proj_net_y - physical_net_y)

                                if net_diff > 0.08 * h:
                                    continue

                                line_mask = np.zeros((h, w), dtype=np.uint8)
                                for p1, p2 in STANDARD_COURT_LINES:
                                    pts = np.array([[p1[0], p1[1]], [p2[0], p2[1]]], dtype=np.float32).reshape(-1, 1, 2)
                                    proj = cv2.perspectiveTransform(pts, H)
                                    pt1 = tuple(np.round(proj[0][0]).astype(int))
                                    pt2 = tuple(np.round(proj[1][0]).astype(int))
                                    cv2.line(line_mask, pt1, pt2, 255, 2)

                                overlap = cv2.bitwise_and(white_mask, line_mask)
                                overlap_pixels = cv2.countNonZero(overlap)

                                if overlap_pixels > best_overlap:
                                    best_overlap = overlap_pixels
                                    best_corners = corners

        if best_corners is not None and best_overlap >= 3000:
            return best_corners

        return None
