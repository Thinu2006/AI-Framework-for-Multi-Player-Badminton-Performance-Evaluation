import cv2
import numpy as np

try:
    from court_geometry import CourtGeometry
except ModuleNotFoundError:
    from court_detection.court_geometry import CourtGeometry


class CourtMapper:
    def __init__(
        self,
        corners_pixel,
        alpha=0.35,
        top_view_width=420,
        top_view_margin=30
    ):
        """
        Parameters
        ----------
        corners_pixel : dict
            Dictionary containing:
                far_left
                far_right
                near_left
                near_right

        alpha : float
            Opacity of the perspective-view zone overlay.

        top_view_width : int
            Width in pixels of the playable court area in the top view.
            Margins are added outside this width.

        top_view_margin : int
            Empty margin around the top-view court.
        """
        self.corners_pixel = corners_pixel
        self.alpha = alpha

        self.geometry = CourtGeometry(corners_pixel)
        self.lines_pixel = self.geometry.get_lines_pixel()
        self.zones_pixel = self.geometry.get_zones()

        self.top_view_width = int(top_view_width)
        self.top_view_margin = int(top_view_margin)

        # Preserve the official court aspect ratio.
        self.top_view_height = int(
            round(
                self.top_view_width
                * self.geometry.LENGTH
                / self.geometry.WIDTH
            )
        )

        self.top_canvas_width = (
            self.top_view_width + 2 * self.top_view_margin
        )
        self.top_canvas_height = (
            self.top_view_height + 2 * self.top_view_margin
        )

        # Court coordinates are centimetres, so the x and y scales
        # should be the same to preserve geometry.
        self.top_scale = (
            self.top_view_width / self.geometry.WIDTH
        )

        # Homography from canonical court coordinates to top-view pixels.
        self.H_court_to_top = np.array(
            [
                [
                    self.top_scale,
                    0.0,
                    self.top_view_margin
                ],
                [
                    0.0,
                    self.top_scale,
                    self.top_view_margin
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            dtype=np.float64
        )

        self.H_top_to_court = np.linalg.inv(
            self.H_court_to_top
        )

        # Camera pixel -> court -> top-view pixel.
        self.H_pixel_to_top = (
            self.H_court_to_top @ self.geometry.H_inv
        )

        # Top-view pixel -> court -> camera pixel.
        self.H_top_to_pixel = np.linalg.inv(
            self.H_pixel_to_top
        )

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_homography(points, matrix):
        """
        Apply a 3x3 homography to one or more 2D points.
        """
        points = np.atleast_2d(points).astype(np.float64)

        homogeneous = np.hstack(
            [
                points,
                np.ones(
                    (len(points), 1),
                    dtype=np.float64
                )
            ]
        )

        transformed = (matrix @ homogeneous.T).T

        denominator = transformed[:, 2:3]

        # Avoid invalid division in degenerate cases.
        denominator[np.abs(denominator) < 1e-9] = 1e-9

        return transformed[:, :2] / denominator

    def court_to_top_view(self, points):
        """
        Convert canonical court coordinates into top-view pixels.
        """
        return self._apply_homography(
            points,
            self.H_court_to_top
        )

    def top_view_to_court(self, points):
        """
        Convert top-view pixels into canonical court coordinates.
        """
        return self._apply_homography(
            points,
            self.H_top_to_court
        )

    def pixel_to_top_view(self, points):
        """
        Convert camera-frame pixels directly into top-view pixels.
        """
        return self._apply_homography(
            points,
            self.H_pixel_to_top
        )

    def top_view_to_pixel(self, points):
        """
        Convert top-view pixels back into camera-frame pixels.
        """
        return self._apply_homography(
            points,
            self.H_top_to_pixel
        )

    def is_inside_court(self, court_point, margin=150.0):
        """
        Check whether a court-coordinate point is inside or near the
        full doubles court (allowing margin in cm).
        """
        x, y = court_point

        return (
            -margin <= x <= self.geometry.WIDTH + margin
            and
            -margin <= y <= self.geometry.LENGTH + margin
        )

    # ------------------------------------------------------------------
    # Perspective camera overlay
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame,
        draw_zones=True,
        draw_lines=True,
        draw_corners=True,
        draw_hud=True
    ):
        """
        Apply the original court overlay to a camera frame.

        This method retains the behaviour of your existing implementation.
        """
        output = frame.copy()

        # 1. Draw semi-transparent zone polygons.
        if draw_zones:
            overlay = output.copy()

            for zone in self.zones_pixel:
                cv2.fillPoly(
                    overlay,
                    [zone['poly_pixel']],
                    zone['color']
                )

            cv2.addWeighted(
                overlay,
                self.alpha,
                output,
                1.0 - self.alpha,
                0,
                output
            )

            for zone in self.zones_pixel:
                moments = cv2.moments(zone['poly_pixel'])

                if moments["m00"] == 0:
                    continue

                centre_x = int(
                    moments["m10"] / moments["m00"]
                )
                centre_y = int(
                    moments["m01"] / moments["m00"]
                )

                label = zone['name']
                font_scale = 0.42
                thickness = 1

                (text_width, text_height), _ = (
                    cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        thickness
                    )
                )

                background_point_1 = (
                    centre_x - text_width // 2 - 5,
                    centre_y - text_height // 2 - 4
                )
                background_point_2 = (
                    centre_x + text_width // 2 + 5,
                    centre_y + text_height // 2 + 4
                )

                text_overlay = output.copy()

                cv2.rectangle(
                    text_overlay,
                    background_point_1,
                    background_point_2,
                    (15, 15, 20),
                    -1
                )

                cv2.addWeighted(
                    text_overlay,
                    0.75,
                    output,
                    0.25,
                    0,
                    output
                )

                cv2.rectangle(
                    output,
                    background_point_1,
                    background_point_2,
                    zone['color'],
                    1,
                    cv2.LINE_AA
                )

                text_point = (
                    centre_x - text_width // 2,
                    centre_y + text_height // 2 - 1
                )

                cv2.putText(
                    output,
                    label,
                    text_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

        # 2. Draw court lines.
        if draw_lines:
            for name, (point_1, point_2) in (
                self.lines_pixel.items()
            ):
                is_outer = 'outer' in name
                is_net = 'net' in name

                if is_outer:
                    colour = (0, 255, 255)
                    thickness = 3
                elif is_net:
                    colour = (0, 255, 0)
                    thickness = 3
                else:
                    colour = (240, 240, 240)
                    thickness = 2

                cv2.line(
                    output,
                    point_1,
                    point_2,
                    colour,
                    thickness,
                    cv2.LINE_AA
                )

                if is_net:
                    middle_x = (
                        point_1[0] + point_2[0]
                    ) // 2
                    middle_y = (
                        point_1[1] + point_2[1]
                    ) // 2

                    net_label = "NET"

                    (label_width, label_height), _ = (
                        cv2.getTextSize(
                            net_label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            2
                        )
                    )

                    cv2.rectangle(
                        output,
                        (
                            middle_x - label_width // 2 - 10,
                            middle_y - label_height // 2 - 6
                        ),
                        (
                            middle_x + label_width // 2 + 10,
                            middle_y + label_height // 2 + 6
                        ),
                        (0, 180, 0),
                        -1
                    )

                    cv2.putText(
                        output,
                        net_label,
                        (
                            middle_x - label_width // 2,
                            middle_y + label_height // 2 - 2
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

        # 3. Draw calibrated corner points.
        if draw_corners:
            corner_colours = {
                'far_left': (0, 0, 255),
                'far_right': (0, 0, 255),
                'near_left': (0, 255, 0),
                'near_right': (0, 255, 0)
            }

            for name, coordinate in self.corners_pixel.items():
                x, y = coordinate
                integer_x = int(round(x))
                integer_y = int(round(y))

                colour = corner_colours.get(
                    name,
                    (0, 0, 255)
                )

                cv2.circle(
                    output,
                    (integer_x, integer_y),
                    7,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA
                )

                cv2.circle(
                    output,
                    (integer_x, integer_y),
                    5,
                    colour,
                    -1,
                    cv2.LINE_AA
                )

                label = (
                    f"{name}: ({x:.1f}, {y:.1f})"
                )

                offset_y = (
                    -12 if "far" in name else 22
                )
                offset_x = (
                    10 if "right" in name else -120
                )

                text_position = (
                    integer_x + offset_x,
                    integer_y + offset_y
                )

                (text_width, text_height), _ = (
                    cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        1
                    )
                )

                cv2.rectangle(
                    output,
                    (
                        text_position[0] - 4,
                        text_position[1] - text_height - 4
                    ),
                    (
                        text_position[0] + text_width + 4,
                        text_position[1] + 4
                    ),
                    (20, 20, 20),
                    -1
                )

                cv2.rectangle(
                    output,
                    (
                        text_position[0] - 4,
                        text_position[1] - text_height - 4
                    ),
                    (
                        text_position[0] + text_width + 4,
                        text_position[1] + 4
                    ),
                    colour,
                    1
                )

                cv2.putText(
                    output,
                    label,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # 4. Draw HUD.
        if draw_hud:
            hud = output.copy()

            cv2.rectangle(
                hud,
                (20, 20),
                (360, 95),
                (15, 15, 25),
                -1
            )

            cv2.addWeighted(
                hud,
                0.75,
                output,
                0.25,
                0,
                output
            )

            cv2.rectangle(
                output,
                (20, 20),
                (360, 95),
                (0, 255, 200),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                output,
                "COURT MAPPING SYSTEM",
                (32, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                output,
                "Model: Standard BWF Court (13.4m x 6.1m)",
                (32, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                output,
                "Method: Homography Perspective Calibration",
                (32, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )

        return output

