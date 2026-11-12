import cv2
import numpy as np

class CourtGeometry:
    """
    Badminton Court Geometry and Homography Mapping System.
    
    Standard BWF Court Dimensions (in centimeters):
    - Width: 610 cm (6.10 m)
    - Length: 1340 cm (13.40 m)
    - Singles Side Alley: 46 cm each side (Singles width = 518 cm)
    - Back Service Line (Doubles): 76 cm from back boundary
    - Short Service Line: 198 cm from net (Net at 670 cm)
      -> Far Short Service Line: 670 - 198 = 472 cm
      -> Near Short Service Line: 670 + 198 = 868 cm
      -> Far Doubles Service Line: 76 cm
      -> Near Doubles Service Line: 1340 - 76 = 1264 cm
    - Center Line x = 305 cm
    """

    WIDTH = 610.0
    LENGTH = 1340.0

    def __init__(self, corners_pixel):
        """
        corners_pixel: dict with keys 'far_left', 'far_right', 'near_left', 'near_right'
        """
        self.corners_pixel = corners_pixel
        
        # Source 2D points (standard court space)
        self.pts_2d = np.array([
            [0.0, 0.0],          # far_left
            [self.WIDTH, 0.0],   # far_right
            [self.WIDTH, self.LENGTH], # near_right
            [0.0, self.LENGTH]   # near_left
        ], dtype=np.float32)

        # Destination pixel points
        self.pts_pixel = np.array([
            corners_pixel['far_left'],
            corners_pixel['far_right'],
            corners_pixel['near_right'],
            corners_pixel['near_left']
        ], dtype=np.float32)

        # Compute Homography Matrix H: 2D court -> Pixel frame
        self.H, _ = cv2.findHomography(self.pts_2d, self.pts_pixel)
        # Inverse Homography Matrix H_inv: Pixel frame -> 2D court
        self.H_inv = np.linalg.inv(self.H)

    def court_to_pixel(self, pts):
        """
        Transform (N, 2) array of 2D court points into pixel frame coordinates.
        """
        pts = np.atleast_2d(pts).astype(np.float32)
        pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        transformed = (self.H @ pts_homo.T).T
        pixel_pts = transformed[:, :2] / transformed[:, 2:]
        return pixel_pts

    def pixel_to_court(self, pts):
        """
        Transform (N, 2) array of pixel frame coordinates into 2D court coordinates.
        """
        pts = np.atleast_2d(pts).astype(np.float32)
        pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        transformed = (self.H_inv @ pts_homo.T).T
        court_pts = transformed[:, :2] / transformed[:, 2:]
        return court_pts

    def get_lines_pixel(self):
        """
        Returns dictionary of all court lines transformed into pixel coordinates.
        """
        lines_2d = {
            'outer_far_boundary': [[0, 0], [610, 0]],
            'outer_near_boundary': [[0, 1340], [610, 1340]],
            'outer_left_boundary': [[0, 0], [0, 1340]],
            'outer_right_boundary': [[610, 0], [610, 1340]],
            
            'singles_left_sideline': [[46, 0], [46, 1340]],
            'singles_right_sideline': [[564, 0], [564, 1340]],
            
            'far_doubles_service_line': [[0, 76], [610, 76]],
            'near_doubles_service_line': [[0, 1264], [610, 1264]],
            
            'far_short_service_line': [[0, 472], [610, 472]],
            'near_short_service_line': [[0, 868], [610, 868]],
            
            'far_rear_mid_line': [[0, 160], [610, 160]],
            'far_mid_front_line': [[0, 320], [610, 320]],
            'near_front_mid_line': [[0, 1020], [610, 1020]],
            'near_mid_rear_line': [[0, 1180], [610, 1180]],
            
            'net_line': [[0, 670], [610, 670]],
            
            'far_center_line': [[305, 0], [305, 472]],
            'near_center_line': [[305, 868], [305, 1340]],
        }

        lines_pixel = {}
        for name, pts in lines_2d.items():
            pixel_pts = self.court_to_pixel(pts)
            lines_pixel[name] = [(int(round(p[0])), int(round(p[1]))) for p in pixel_pts]

        return lines_pixel

    def get_zones(self):
        """
        Returns list of dictionary definitions for court zones in 2D space & transformed pixel space.
        Zone names follow the user layout:
          - Far Right | Far Left
          - NET
          - Near Left | Near Right
        """
        zones_def = [
            # --- FAR COURT ZONES ---
            {
                'name': 'Far Right Rear',
                'color': (255, 100, 0), # Deep Orange
                'poly_2d': [[0, 0], [305, 0], [305, 160], [0, 160]]
            },
            {
                'name': 'Far Left Rear',
                'color': (200, 30, 160), # Magenta
                'poly_2d': [[305, 0], [610, 0], [610, 160], [305, 160]]
            },
            {
                'name': 'Far Right Mid',
                'color': (255, 160, 50), # Orange-Yellow
                'poly_2d': [[0, 160], [305, 160], [305, 320], [0, 320]]
            },
            {
                'name': 'Far Left Mid',
                'color': (230, 80, 200), # Light Purple
                'poly_2d': [[305, 160], [610, 160], [610, 320], [305, 320]]
            },
            {
                'name': 'Far Right Front',
                'color': (255, 210, 80), # Yellow
                'poly_2d': [[0, 320], [305, 320], [305, 670], [0, 670]]
            },
            {
                'name': 'Far Left Front',
                'color': (255, 140, 230), # Pink
                'poly_2d': [[305, 320], [610, 320], [610, 670], [305, 670]]
            },

            # --- NEAR COURT ZONES ---
            {
                'name': 'Near Left Front',
                'color': (80, 240, 200), # Turquoise
                'poly_2d': [[0, 670], [305, 670], [305, 1020], [0, 1020]]
            },
            {
                'name': 'Near Right Front',
                'color': (80, 200, 255), # Sky Blue
                'poly_2d': [[305, 670], [610, 670], [610, 1020], [305, 1020]]
            },
            {
                'name': 'Near Left Mid',
                'color': (50, 205, 50), # Lime Green
                'poly_2d': [[0, 1020], [305, 1020], [305, 1180], [0, 1180]]
            },
            {
                'name': 'Near Right Mid',
                'color': (30, 144, 255), # Royal Blue
                'poly_2d': [[305, 1020], [610, 1020], [610, 1180], [305, 1180]]
            },
            {
                'name': 'Near Left Rear',
                'color': (0, 160, 100), # Dark Emerald
                'poly_2d': [[0, 1180], [305, 1180], [305, 1340], [0, 1340]]
            },
            {
                'name': 'Near Right Rear',
                'color': (0, 90, 220), # Deep Blue
                'poly_2d': [[305, 1180], [610, 1180], [610, 1340], [305, 1340]]
            }
        ]

        zones_pixel = []
        for zone in zones_def:
            pts_pix = self.court_to_pixel(zone['poly_2d'])
            poly_pix = np.array(pts_pix, dtype=np.int32)
            zones_pixel.append({
                'name': zone['name'],
                'color': zone['color'],
                'poly_2d': zone['poly_2d'],
                'poly_pixel': poly_pix
            })

        return zones_pixel
    def get_lines_court(self):
        """
        Return all court lines in canonical court coordinates.

        Coordinates are expressed in centimetres:
            x: 0 to 610
            y: 0 to 1340
        """
        return {
            'outer_far_boundary': [[0, 0], [610, 0]],
            'outer_near_boundary': [[0, 1340], [610, 1340]],
            'outer_left_boundary': [[0, 0], [0, 1340]],
            'outer_right_boundary': [[610, 0], [610, 1340]],

            'singles_left_sideline': [[46, 0], [46, 1340]],
            'singles_right_sideline': [[564, 0], [564, 1340]],

            'far_doubles_service_line': [[0, 76], [610, 76]],
            'near_doubles_service_line': [[0, 1264], [610, 1264]],

            'far_short_service_line': [[0, 472], [610, 472]],
            'near_short_service_line': [[0, 868], [610, 868]],

            'far_rear_mid_line': [[0, 160], [610, 160]],
            'far_mid_front_line': [[0, 320], [610, 320]],
            'near_front_mid_line': [[0, 1020], [610, 1020]],
            'near_mid_rear_line': [[0, 1180], [610, 1180]],

            'net_line': [[0, 670], [610, 670]],

            'far_center_line': [[305, 0], [305, 472]],
            'near_center_line': [[305, 868], [305, 1340]],
        }
    def get_lines_pixel(self):
        """
        Return all court lines transformed into camera-frame pixels.
        """
        lines_pixel = {}

        for name, pts in self.get_lines_court().items():
            pixel_pts = self.court_to_pixel(pts)
            lines_pixel[name] = [
                (int(round(point[0])), int(round(point[1])))
                for point in pixel_pts
            ]

        return lines_pixel
