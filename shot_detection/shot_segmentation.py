class ShuttleSegmenter:
    """
    Buffers shuttle detections & player zones frame-by-frame,
    grouping shuttle flight into single shot segments with reversal splitting.
    Supports both 2D court metric coordinates (cm) and pixel coordinates.
    """

    def __init__(self, max_missing_gap: int = 4, min_direction_consistency: float = 0.70, min_valid_points: int = 6):
        self.max_missing_gap = max_missing_gap
        self.min_direction_consistency = min_direction_consistency
        self.min_valid_points = min_valid_points
        self.reset()

    def reset(self):
        self.is_active = False
        self.start_frame = None
        self.records = []
        self.missing_counter = 0
        self.reversal_records = []
        self.primary_dir = None

    def add_frame(
        self,
        frame_idx: int,
        shuttle_pos: tuple,
        player_zones: dict,
        shuttle_court_pos: tuple = None
    ) -> dict:
        """
        Adds a frame to tracking buffer. Returns a completed segment dict if segment ended, else None.
        Handles reversal splitting cleanly between consecutive flights.
        """
        # Select active position: preference given to 2D court metric coordinate (cm)
        active_pos = shuttle_court_pos if shuttle_court_pos is not None else shuttle_pos
        is_metric = (shuttle_court_pos is not None)

        dy_min_start = 15.0 if is_metric else 3.0
        reversal_delta = 20.0 if is_metric else 4.0

        # Case 1: Inactive segment
        if not self.is_active:
            if active_pos is not None:
                self.records.append({
                    'frame_idx': frame_idx,
                    'shuttle_pos': shuttle_pos,
                    'shuttle_court_pos': shuttle_court_pos,
                    'active_pos': active_pos,
                    'player_zones': player_zones
                })
                if len(self.records) >= 2:
                    valid_pts = [r['active_pos'] for r in self.records if r['active_pos'] is not None]
                    if len(valid_pts) >= 2:
                        dy = valid_pts[-1][1] - valid_pts[-2][1]
                        if abs(dy) >= dy_min_start:
                            self.is_active = True
                            self.start_frame = self.records[0]['frame_idx']
                            self.primary_dir = 'far_to_near' if dy > 0 else 'near_to_far'
                            self.missing_counter = 0
                            self.reversal_records = []
                else:
                    if len(self.records) > 5:
                        self.records.pop(0)
            return None

        # Case 2: Active segment - Missing shuttle tracking
        if active_pos is None:
            self.missing_counter += 1
            self.records.append({
                'frame_idx': frame_idx,
                'shuttle_pos': None,
                'shuttle_court_pos': None,
                'active_pos': None,
                'player_zones': player_zones
            })
            if self.missing_counter > self.max_missing_gap:
                return self._finalize_segment(reason="max_missing_gap_exceeded")
            return None
        else:
            self.missing_counter = 0

        # Check for direction reversal using Y coordinate extremum of valid shuttle points in current segment
        valid_records = [r for r in self.records if r['active_pos'] is not None]
        if len(valid_records) >= 2:
            if self.primary_dir == 'far_to_near':
                max_y_seen = max(r['active_pos'][1] for r in valid_records)
                is_reversal = (active_pos[1] < max_y_seen - reversal_delta)
            elif self.primary_dir == 'near_to_far':
                min_y_seen = min(r['active_pos'][1] for r in valid_records)
                is_reversal = (active_pos[1] > min_y_seen + reversal_delta)
            else:
                is_reversal = False

            if is_reversal:
                self.reversal_records.append({
                    'frame_idx': frame_idx,
                    'shuttle_pos': shuttle_pos,
                    'shuttle_court_pos': shuttle_court_pos,
                    'active_pos': active_pos,
                    'player_zones': player_zones
                })
            else:
                self.reversal_records = []

        self.records.append({
            'frame_idx': frame_idx,
            'shuttle_pos': shuttle_pos,
            'shuttle_court_pos': shuttle_court_pos,
            'active_pos': active_pos,
            'player_zones': player_zones
        })

        # Close segment if 2 consecutive reversal frames detected
        if len(self.reversal_records) >= 2:
            return self._finalize_segment(reason="direction_reversal")

        return None

    def force_finalize(self) -> dict:
        if self.is_active:
            return self._finalize_segment(reason="forced_end")
        return None

    def _finalize_segment(self, reason: str = "") -> dict:
        """
        Splits reversal frames cleanly, validates segment rules (min 6 points, net crossing),
        and resets segmenter state for next flight.
        """
        records_snapshot = list(self.records)
        reversal_snapshot = list(self.reversal_records)
        primary_dir = self.primary_dir

        # Reversal splitting: exclude reversal frames from current segment
        if reason == "direction_reversal" and len(reversal_snapshot) >= 2:
            cut_len = len(reversal_snapshot)
            current_segment_records = records_snapshot[:-cut_len]
        else:
            current_segment_records = records_snapshot

        self.reset()

        # Seed next segment with reversal frames if reversal occurred
        if reason == "direction_reversal" and len(reversal_snapshot) >= 2:
            self.records = list(reversal_snapshot)
            if len(self.records) >= 2:
                valid_pts = [r['active_pos'] for r in self.records if r['active_pos'] is not None]
                if len(valid_pts) >= 2:
                    dy = valid_pts[-1][1] - valid_pts[-2][1]
                    self.is_active = True
                    self.start_frame = self.records[0]['frame_idx']
                    self.primary_dir = 'far_to_near' if dy > 0 else 'near_to_far'

        valid_records = [r for r in current_segment_records if r['active_pos'] is not None]

        # Requirement 6: Require at least 6 valid shuttle points
        if len(valid_records) < self.min_valid_points:
            return None

        # Direction consistency check (>= 70%)
        match_count = 0
        total_pairs = len(valid_records) - 1
        for i in range(total_pairs):
            dy = valid_records[i+1]['active_pos'][1] - valid_records[i]['active_pos'][1]
            if (primary_dir == 'far_to_near' and dy >= 0) or \
               (primary_dir == 'near_to_far' and dy <= 0):
                match_count += 1

        direction_consistency = match_count / float(total_pairs) if total_pairs > 0 else 0.0
        if direction_consistency < self.min_direction_consistency:
            return None

        # Net Crossing & Trajectory Distance Validation
        start_y = valid_records[0]['active_pos'][1]
        end_y = valid_records[-1]['active_pos'][1]
        dy_total = end_y - start_y
        is_metric = (valid_records[0].get('shuttle_court_pos') is not None)
        min_dy_total = 150.0 if is_metric else 35.0

        if primary_dir == 'far_to_near':
            if dy_total < min_dy_total:  # Shuttle must travel downward across net/court
                return None
        elif primary_dir == 'near_to_far':
            if dy_total > -min_dy_total:  # Shuttle must travel upward across net/court
                return None

        return {
            'records': current_segment_records,
            'valid_records': valid_records,
            'primary_dir': primary_dir,
            'direction_consistency': direction_consistency,
            'reason': reason
        }


