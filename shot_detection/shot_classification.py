import math
from collections import Counter
import numpy as np
import cv2
from .shot_segmentation import ShuttleSegmenter

def normalize_zone_name(zone_str: str) -> str:
    """
    Normalizes court zone names (e.g., 'Far Left Rear' -> 'far_left_rear').
    Returns 'unknown' for invalid/unassigned zones.
    """
    if not zone_str or not isinstance(zone_str, str):
        return "unknown"
    clean = zone_str.strip().lower()
    if clean in ["court", "unknown", "none", "n/a", ""]:
        return "unknown"
    return clean.replace(" ", "_")


def validate_player_role_zone(role: str, zone_name: str) -> str:
    """
    Validates that Far Player only has far_* zones and Near Player only has near_* zones.
    Returns normalized zone string if valid, else 'unknown'.
    """
    norm_zone = normalize_zone_name(zone_name)
    if norm_zone == "unknown":
        return "unknown"

    role_clean = role.lower().strip()
    if "far" in role_clean:
        if not norm_zone.startswith("far_"):
            return "unknown"
    elif "near" in role_clean:
        if not norm_zone.startswith("near_"):
            return "unknown"

    return norm_zone


def get_zone_at_pixel(geometry, pixel_coord: tuple) -> str:
    """
    Given a pixel coordinate (x, y), projects into 2D court space.
    Returns normalized zone name if point lies inside or is nearest to a tactical zone.
    """
    if geometry is None or pixel_coord is None:
        return "unknown"
    try:
        court_pts = geometry.pixel_to_court([pixel_coord])
        cx, cy = float(court_pts[0][0]), float(court_pts[0][1])

        best_zone = "unknown"
        max_dist = -float('inf')

        for zone in geometry.get_zones():
            poly_2d = np.array(zone["poly_2d"], dtype=np.float32)
            dist = cv2.pointPolygonTest(poly_2d, (cx, cy), measureDist=True)
            if dist >= 0:
                return normalize_zone_name(zone["name"])
            
            # Keep track of the nearest zone if point is outside (dist is negative)
            if dist > max_dist:
                max_dist = dist
                best_zone = normalize_zone_name(zone["name"])
                
        # If the point is outside but reasonably close (e.g. within 300cm), return the closest zone
        if max_dist > -300.0:
            return best_zone
            
        return "unknown"
    except Exception:
        return "unknown"


# Zone Category Constants
FAR_REAR_ZONES = {"far_right_rear", "far_left_rear"}
NEAR_REAR_ZONES = {"near_left_rear", "near_right_rear"}
FAR_MID_ZONES = {"far_right_mid", "far_left_mid"}
NEAR_MID_ZONES = {"near_left_mid", "near_right_mid"}
FAR_FRONT_ZONES = {"far_right_front", "far_left_front"}
NEAR_FRONT_ZONES = {"near_left_front", "near_right_front"}


class ShotClassifier:
    """
    Classifies badminton shot types using player tactical zones as primary,
    shuttle target zone as secondary fallback, and trajectory tie-breakers.
    """

    @staticmethod
    def classify_by_zones(hitter_zone: str, receiver_or_target_zone: str) -> str:
        """
        Applies exact zone rules for CLEAR, DROP, SMASH, NET SHOT, LIFT.
        Rule table:
          Rear -> Rear   = Clear
          Rear -> Front  = Drop
          Rear -> Mid    = Smash
          Front -> Front = Net Shot
          Front -> Rear  = Lift
          Otherwise       = None
        """
        hz = normalize_zone_name(hitter_zone)
        tz = normalize_zone_name(receiver_or_target_zone)

        if hz == "unknown" or tz == "unknown":
            return None

        # Enforce that hitter and target/receiver are on opposite court sides
        is_far_hitter = hz.startswith("far_")
        is_near_hitter = hz.startswith("near_")
        is_far_target = tz.startswith("far_")
        is_near_target = tz.startswith("near_")

        if (is_far_hitter and not is_near_target) or (is_near_hitter and not is_far_target):
            return None

        is_hitter_rear_or_mid = (hz in FAR_REAR_ZONES) or (hz in NEAR_REAR_ZONES) or (hz in FAR_MID_ZONES) or (hz in NEAR_MID_ZONES)
        is_hitter_front = (hz in FAR_FRONT_ZONES) or (hz in NEAR_FRONT_ZONES)

        is_target_rear = (tz in NEAR_REAR_ZONES) or (tz in FAR_REAR_ZONES)
        is_target_front = (tz in NEAR_FRONT_ZONES) or (tz in FAR_FRONT_ZONES)
        is_target_mid = (tz in NEAR_MID_ZONES) or (tz in FAR_MID_ZONES)

        if is_hitter_rear_or_mid:
            if is_target_rear:
                return "overhead_clear"
            elif is_target_front:
                return "drop"
            elif is_target_mid:
                return "smash"
        elif is_hitter_front:
            if is_target_front or is_target_mid:
                return "net_shot"
            elif is_target_rear:
                return "lift"

        return None

    @staticmethod
    def classify_by_trajectory(hitter_zone: str, target_zone: str, trajectory_pts: list) -> str:
        """
        Tie-breaker using shuttle trajectory dynamics.
        """
        hz = normalize_zone_name(hitter_zone)
        tz = normalize_zone_name(target_zone)

        is_hitter_rear_or_mid = hz in (FAR_REAR_ZONES | NEAR_REAR_ZONES | FAR_MID_ZONES | NEAR_MID_ZONES)
        is_hitter_front = hz in (FAR_FRONT_ZONES | NEAR_FRONT_ZONES)
        is_target_rear = tz in (FAR_REAR_ZONES | NEAR_REAR_ZONES)
        is_target_mid = tz in (FAR_MID_ZONES | NEAR_MID_ZONES)
        is_target_front = tz in (FAR_FRONT_ZONES | NEAR_FRONT_ZONES)

        if len(trajectory_pts) >= 3:
            pts = np.array(trajectory_pts)
            diffs = np.diff(pts, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            avg_speed = float(np.mean(distances))
            
            x = pts[:, 0]
            y = pts[:, 1]
            if len(pts) >= 4 and np.max(x) - np.min(x) > 5:
                try:
                    poly = np.polyfit(x, y, 2)
                    curvature = abs(poly[0])
                except Exception:
                    curvature = 0.0
            else:
                curvature = 0.0
        else:
            avg_speed = 10.0
            curvature = 0.0

        if is_hitter_rear_or_mid:
            if is_target_rear or (not is_target_front and curvature > 0.001):
                return "clear"
            if is_target_front or avg_speed < 8.0:
                return "drop"
            if is_target_mid or avg_speed >= 8.0:
                return "smash"
        elif is_hitter_front:
            if is_target_front or avg_speed < 8.0:
                return "net_shot"
            if is_target_rear or curvature > 0.0005:
                return "lift"

        return "unknown"

    @classmethod
    def determine_shot(
        cls,
        hitter_zone: str,
        receiver_zone: str,
        shuttle_target_zone: str,
        trajectory_pts: list
    ) -> tuple:
        """
        Primary Classification Priority:
          1. hitter_zone + receiver_zone (Player Tactical Zones)
          2. hitter_zone + shuttle_target_zone (Secondary Fallback)
          3. Trajectory dynamics tie-breaker
          4. Default to 'unknown'
        Returns: (shot_type, confidence_rule_quality)
        """
        # Priority 1: Player Tactical Zones (hitter_zone + receiver_zone)
        if hitter_zone != "unknown" and receiver_zone != "unknown":
            shot = cls.classify_by_zones(hitter_zone, receiver_zone)
            if shot is not None:
                return shot, 1.0

        # Priority 2: Secondary Fallback (hitter_zone + shuttle_target_zone)
        if hitter_zone != "unknown" and shuttle_target_zone != "unknown":
            shot = cls.classify_by_zones(hitter_zone, shuttle_target_zone)
            if shot is not None:
                return shot, 0.75

        # Priority 3: Trajectory Tie-Breaker
        effective_target = receiver_zone if receiver_zone != "unknown" else shuttle_target_zone
        shot = cls.classify_by_trajectory(hitter_zone, effective_target, trajectory_pts)
        if shot != "unknown":
            return shot, 0.60

        return "unknown", 0.0


class BadmintonShotDetector:
    """
    Main manager for Badminton Shot Detection system.
    Processes frame-by-frame inputs (in camera pixels & 2D top-view court space)
    and outputs prediction JSON dictionaries.
    """

    def __init__(self, geometry=None, max_missing_gap: int = 4, min_direction_consistency: float = 0.70, min_valid_points: int = 6):
        self.geometry = geometry
        self.segmenter = ShuttleSegmenter(
            max_missing_gap=max_missing_gap,
            min_direction_consistency=min_direction_consistency,
            min_valid_points=min_valid_points
        )
        self.completed_shots = []
        self.shot_counter = 0

    def set_geometry(self, geometry):
        self.geometry = geometry

    def process_frame(
        self,
        frame_idx: int,
        shuttle_pos: tuple,
        player_zones: dict,
        shuttle_court_pos: tuple = None
    ) -> dict:
        norm_player_zones = {
            role: validate_player_role_zone(role, zname) for role, zname in player_zones.items()
        }

        # Convert to 2D metric court space if geometry is available and court pos not provided
        if shuttle_court_pos is None and shuttle_pos is not None and self.geometry is not None:
            try:
                c_pts = self.geometry.pixel_to_court([shuttle_pos])
                shuttle_court_pos = (float(c_pts[0][0]), float(c_pts[0][1]))
            except Exception:
                shuttle_court_pos = None

        segment_data = self.segmenter.add_frame(
            frame_idx=frame_idx,
            shuttle_pos=shuttle_pos,
            player_zones=norm_player_zones,
            shuttle_court_pos=shuttle_court_pos
        )
        if segment_data is not None:
            shot_dict = self._build_shot_prediction(segment_data)
            if shot_dict is not None:
                self.completed_shots.append(shot_dict)
                return shot_dict
        return None

    def finish_stream(self) -> list:
        segment_data = self.segmenter.force_finalize()
        if segment_data is not None:
            shot_dict = self._build_shot_prediction(segment_data)
            if shot_dict is not None:
                self.completed_shots.append(shot_dict)
        return self.completed_shots

    def _get_dominant_zone(self, records: list, role: str) -> tuple:
        """
        Calculates mode of player zone for given role in records window (ignoring 'unknown').
        Validates role-zone compatibility.
        """
        zones = [r['player_zones'].get(role, 'unknown') for r in records if role in r['player_zones']]
        valid_zones = [validate_player_role_zone(role, z) for z in zones]
        valid_zones = [z for z in valid_zones if z != 'unknown']

        if not valid_zones:
            return "unknown", 0.0

        counts = Counter(valid_zones)
        dominant, mode_count = counts.most_common(1)[0]
        stability = mode_count / float(len(records)) if records else 0.0
        return dominant, stability

    def _build_shot_prediction(self, segment_data: dict) -> dict:
        records = segment_data['records']
        valid_records = segment_data['valid_records']
        travel_dir = segment_data['primary_dir']
        dir_consistency = segment_data['direction_consistency']

        start_frame = records[0]['frame_idx']
        end_frame = records[-1]['frame_idx']

        # Requirement 2: Assign hitter & receiver based on travel direction
        if travel_dir == 'far_to_near':
            hitter_role = 'Far Player'
            receiver_role = 'Near Player'
            hitter_str = 'far_player'
            receiver_str = 'near_player'
        else:
            hitter_role = 'Near Player'
            receiver_role = 'Far Player'
            hitter_str = 'near_player'
            receiver_str = 'far_player'

        # Requirement 3: Event-windowed dominant zones (3-5 frames)
        start_window = records[:min(5, len(records))]
        end_window = records[-min(5, len(records)):]

        hitter_zone, hitter_stability = self._get_dominant_zone(start_window, hitter_role)
        receiver_zone, receiver_stability = self._get_dominant_zone(end_window, receiver_role)

        # Requirement 5: Shuttle target zone via inverse homography (strict polygon check)
        final_shuttle_pos = valid_records[-1]['shuttle_pos']
        shuttle_target_zone = get_zone_at_pixel(self.geometry, final_shuttle_pos)

        traj_pts = [r['shuttle_pos'] for r in valid_records]

        # Requirement 4 & 8: Classify shot prioritizing player zones
        shot_type, rule_quality = ShotClassifier.determine_shot(
            hitter_zone=hitter_zone,
            receiver_zone=receiver_zone,
            shuttle_target_zone=shuttle_target_zone,
            trajectory_pts=traj_pts
        )

        # Determine forehand vs backhand side based on player zone (right = forehand, left = backhand)
        is_right_zone = "right" in hitter_zone.lower()
        is_rear_or_mid_zone = (hitter_zone in FAR_REAR_ZONES) or (hitter_zone in NEAR_REAR_ZONES) or (hitter_zone in FAR_MID_ZONES) or (hitter_zone in NEAR_MID_ZONES) or ("rear" in hitter_zone.lower()) or ("mid" in hitter_zone.lower())

        # Format shot_type: smash -> forehand_smash, clear -> forehand_clear, drop -> forehand_drop, net_shot / lift prefixed by side
        if "smash" in shot_type or (is_rear_or_mid_zone and "smash" in shot_type):
            shot_type = "forehand_smash"
        elif "clear" in shot_type or (is_rear_or_mid_zone and "clear" in shot_type):
            shot_type = "forehand_clear"
        elif "drop" in shot_type or (is_rear_or_mid_zone and "drop" in shot_type):
            shot_type = "forehand_drop"
        elif shot_type in ("net_shot", "net shot"):
            shot_type = "forehand_net_shot" if is_right_zone else "backhand_net_shot"
        elif shot_type in ("lift", "net_lift"):
            shot_type = "forehand_net_lift" if is_right_zone else "backhand_net_lift"

        zone_stability = (hitter_stability + receiver_stability) / 2.0
        frame_score = min(1.0, len(valid_records) / 10.0)

        confidence = (
            0.40 * rule_quality +
            0.25 * zone_stability +
            0.20 * dir_consistency +
            0.15 * frame_score
        )
        confidence = round(float(np.clip(confidence, 0.0, 1.0)), 2)

        target_effective = receiver_zone if receiver_zone != "unknown" else shuttle_target_zone
        zone_transition = f"{hitter_zone} -> {target_effective}"

        self.shot_counter += 1

        # Return JSON dict with zone_transition, shuttle_target_zone, and shot_id
        return {
            "shot_id": self.shot_counter,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "travel_direction": travel_dir,
            "hitter": hitter_str,
            "hitter_zone": hitter_zone,
            "receiver": receiver_str,
            "receiver_zone": receiver_zone,
            "shuttle_target_zone": shuttle_target_zone,
            "shuttle_target_method": "approximate_ground_projection",
            "zone_transition": zone_transition,
            "shot_type": shot_type,
            "confidence": confidence
        }

