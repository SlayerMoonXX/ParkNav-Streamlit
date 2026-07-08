"""Expert system for intelligent parking slot recommendation.

Uses a rule-based approach to score and rank available parking slots
based on vehicle type, distance, floor preference, proximity to
elevators/exits, and accessibility requirements.
"""

import os
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from .parking_map import ParkingMap, CellType
from .heuristic import manhattan_3d


@dataclass
class SlotRecommendation:
    """A single parking slot recommendation with score and reasons.

    Attributes:
        slot_id: Unique identifier for the slot (e.g. 'L1-A01').
        floor: Floor index (0-based).
        row: Row position in the grid.
        col: Column position in the grid.
        score: Normalized score between 0.0 and 1.0 (higher is better).
        reasons: Human-readable reasons why this slot was recommended.
        slot_type: Size/type of the slot ('standard', 'large', 'small').
        distance_to_entrance: Manhattan distance from entrance, or None.
    """
    slot_id: str
    floor: int
    row: int
    col: int
    score: float
    reasons: List[str]
    slot_type: str
    distance_to_entrance: Optional[float] = None


class ParkingExpertSystem:
    """Rule-based expert system for recommending optimal parking slots.

    Loads scoring rules from a JSON configuration file and applies them
    to rank available slots based on multiple criteria.
    """

    def __init__(self, rules_path: Optional[str] = None):
        """Initialize the expert system with rules.

        Args:
            rules_path: Path to the expert_rules.json file. If None,
                        uses the default path in models/ directory.
        """
        if rules_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            rules_path = os.path.join(base_dir, 'models', 'expert_rules.json')

        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config = self._default_config()

        self.rules = config.get('rules', [])
        self.scoring = config.get('scoring', {})
        self.vehicle_types = config.get('vehicle_types', {})

    @staticmethod
    def _default_config() -> dict:
        """Return default configuration if rules file is unavailable."""
        return {
            'rules': [],
            'scoring': {
                'distance_weight': 0.3,
                'floor_weight': 0.25,
                'elevator_weight': 0.2,
                'exit_weight': 0.15,
                'accessibility_weight': 0.1,
            },
            'vehicle_types': {
                'sedan': {'size': 'standard', 'label': 'Sedan'},
                'suv': {'size': 'large', 'label': 'SUV / Truck'},
                'motorcycle': {'size': 'small', 'label': 'Motor'},
                'disability': {
                    'size': 'standard',
                    'label': 'Disabilitas',
                    'requires_accessible': True,
                },
            },
        }

    # ----- Legacy interface (keyword args) -----

    def recommend_slots(
        self,
        parking_map: ParkingMap,
        vehicle_type: str = 'sedan',
        prefer_floor: Optional[int] = None,
        require_accessible: bool = False,
        top_n: int = 5,
    ) -> List[SlotRecommendation]:
        """Recommend the best available parking slots (legacy keyword interface).

        Args:
            parking_map: The parking structure with current slot states.
            vehicle_type: Type of vehicle ('sedan', 'suv', 'motorcycle', 'disability').
            prefer_floor: Preferred floor index (0-based), or None for any.
            require_accessible: If True, only return accessible slots.
            top_n: Maximum number of recommendations to return.

        Returns:
            List of SlotRecommendation objects, sorted best first.
        """
        prefs = {
            'vehicle_type': vehicle_type,
            'preferred_floor': prefer_floor,
            'near_elevator': False,
            'near_exit': False,
            'accessibility': require_accessible,
        }
        return self.recommend(parking_map, prefs, top_n=top_n)

    # ----- Primary interface (dict-based) -----

    def recommend(
        self,
        parking_map: ParkingMap,
        preferences: Dict[str, Any],
        top_n: int = 5,
    ) -> List[SlotRecommendation]:
        """Recommend the best available parking slots.

        Args:
            parking_map: The parking structure with current slot states.
            preferences: Dict with keys:
                - vehicle_type (str): 'sedan', 'suv', 'motorcycle', 'disability'
                - preferred_floor (int|None): 0-based index, or None
                - near_elevator (bool): prefer slots close to elevator
                - near_exit (bool): prefer slots close to exit
                - accessibility (bool): require accessible slots only
                - prefer_lower_floor (bool): alias for "prefer ground floor"
            top_n: Maximum number of recommendations to return.

        Returns:
            Sorted list of SlotRecommendation (highest score first).
        """
        vehicle_type = preferences.get('vehicle_type', 'sedan')
        # Normalize vehicle_type (UI may send 'disabilitas' or 'disability')
        vt_lower = vehicle_type.lower()
        if vt_lower == 'disabilitas':
            vt_lower = 'disability'
        if vt_lower == 'motor':
            vt_lower = 'motorcycle'

        preferred_floor = preferences.get('preferred_floor', None)
        near_elevator = preferences.get('near_elevator', False)
        near_exit = preferences.get('near_exit', False)
        require_accessible = preferences.get('accessibility', False)
        prefer_lower = preferences.get('prefer_lower_floor', False)

        # If prefer_lower_floor is set and no explicit floor, default to floor 0
        if prefer_lower and preferred_floor is None:
            preferred_floor = 0

        # Get vehicle info
        v_info = self.vehicle_types.get(
            vt_lower,
            self.vehicle_types.get('sedan', {'size': 'standard'}),
        )
        required_size = v_info.get('size', 'standard')
        needs_accessible = require_accessible or v_info.get('requires_accessible', False)

        # Get reference positions
        entrance = parking_map.get_entrance()
        exit_pos = parking_map.get_exit()
        elevator_positions = [
            (elev['floors'][0], elev['row'], elev['col'])
            for elev in parking_map.elevator_positions
            if elev.get('floors')
        ]

        # Collect available slots
        available = parking_map.get_available_slots()
        if not available:
            return []

        scored: List[SlotRecommendation] = []

        for slot in available:
            # If slot is a dict with 'id' key or comes from dict iteration
            if isinstance(slot, dict):
                slot_data = slot
            else:
                # slot might be a slot_id string
                slot_data = parking_map.slots.get(slot, {})
                slot_data['id'] = slot

            # Filter by accessibility
            if needs_accessible and not slot_data.get('accessible', False):
                continue

            # Filter by size compatibility
            slot_size = slot_data.get('size', 'standard')
            if not self._size_compatible(required_size, slot_size):
                continue

            slot_pos = (slot_data['floor'], slot_data['row'], slot_data['col'])
            score, reasons = self._compute_score(
                slot_data, slot_pos, entrance, exit_pos,
                elevator_positions, preferred_floor,
                near_elevator, near_exit,
            )

            dist_to_entrance = None
            if entrance:
                dist_to_entrance = manhattan_3d(entrance, slot_pos)

            scored.append(SlotRecommendation(
                slot_id=slot_data.get('id', 'unknown'),
                floor=slot_data['floor'],
                row=slot_data['row'],
                col=slot_data['col'],
                score=round(score, 4),
                reasons=reasons,
                slot_type=slot_size,
                distance_to_entrance=dist_to_entrance,
            ))

        # Sort by score descending
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_n]

    # ─────────── Internal helpers ───────────

    def _size_compatible(self, required: str, available: str) -> bool:
        """Check if a slot size can accommodate the vehicle."""
        size_order = {'small': 0, 'motorcycle': 0, 'standard': 1, 'large': 2}
        req_level = size_order.get(required, 1)
        avail_level = size_order.get(available, 1)
        # Motorcycle can fit anywhere; large vehicle needs large slot
        if required == 'small' or required == 'motorcycle':
            return True  # motorcycles fit in any slot
        return avail_level >= req_level

    def _compute_score(
        self,
        slot: dict,
        slot_pos: Tuple[int, int, int],
        entrance: Optional[Tuple[int, int, int]],
        exit_pos: Optional[Tuple[int, int, int]],
        elevator_positions: List[Tuple[int, int, int]],
        preferred_floor: Optional[int],
        near_elevator: bool = False,
        near_exit: bool = False,
    ) -> Tuple[float, List[str]]:
        """Compute a normalized score for a parking slot.

        Returns:
            (score, reasons) where score is 0.0–1.0 and reasons is a list of
            human-readable strings explaining the score.
        """
        score = 0.0
        reasons: List[str] = []
        w = self.scoring

        # 1. Distance from entrance (closer = better)
        if entrance:
            dist = manhattan_3d(entrance, slot_pos)
            max_dist = 30.0
            dist_score = max(0.0, 1.0 - dist / max_dist)
            weight = w.get('distance_weight', 0.3)
            score += weight * dist_score
            if dist_score >= 0.7:
                reasons.append(f"Dekat pintu masuk (jarak: {dist:.0f})")
            elif dist_score >= 0.4:
                reasons.append(f"Jarak sedang dari pintu masuk ({dist:.0f})")
            else:
                reasons.append(f"Jauh dari pintu masuk ({dist:.0f})")

        # 2. Floor preference
        floor_weight = w.get('floor_weight', 0.25)
        if preferred_floor is not None:
            floor_diff = abs(slot['floor'] - preferred_floor)
            floor_score = max(0.0, 1.0 - floor_diff / 3.0)
            if floor_diff == 0:
                reasons.append(f"Lantai sesuai preferensi (Lantai {slot['floor'] + 1})")
        else:
            floor_score = max(0.0, 1.0 - slot['floor'] / 3.0)
        score += floor_weight * floor_score

        # 3. Elevator proximity
        if elevator_positions:
            min_elev_dist = min(
                manhattan_3d(slot_pos, (slot['floor'], ep[1], ep[2]))
                for ep in elevator_positions
            )
            elev_score = max(0.0, 1.0 - min_elev_dist / 15.0)
            elev_weight = w.get('elevator_weight', 0.2)
            # Boost elevator weight if user specifically wants near elevator
            if near_elevator:
                elev_weight = max(elev_weight, 0.3)
            score += elev_weight * elev_score
            if near_elevator and elev_score >= 0.6:
                reasons.append(f"Dekat elevator (jarak: {min_elev_dist:.0f})")
        else:
            elev_weight = w.get('elevator_weight', 0.2)
            score += elev_weight * 0.5  # neutral if no elevator

        # 4. Exit proximity
        if exit_pos:
            exit_dist = manhattan_3d(slot_pos, exit_pos)
            exit_score = max(0.0, 1.0 - exit_dist / 25.0)
            exit_weight = w.get('exit_weight', 0.15)
            if near_exit:
                exit_weight = max(exit_weight, 0.25)
            score += exit_weight * exit_score
            if near_exit and exit_score >= 0.6:
                reasons.append(f"Dekat pintu keluar (jarak: {exit_dist:.0f})")

        # 5. Accessibility bonus
        if slot.get('accessible', False):
            acc_weight = w.get('accessibility_weight', 0.1)
            score += acc_weight
            reasons.append("Slot aksesibel ♿")

        # Add general reason if no specific reasons yet
        if not reasons:
            reasons.append("Slot tersedia")

        return score, reasons

    # ─────────── Public helpers ───────────

    def get_vehicle_types(self) -> Dict[str, Dict[str, Any]]:
        """Get available vehicle type definitions."""
        return dict(self.vehicle_types)

    def get_rules_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all active rules."""
        return [
            {
                'id': r.get('id', ''),
                'name': r.get('name', ''),
                'description': r.get('description', ''),
                'weight': r.get('weight', 0),
            }
            for r in self.rules
        ]

    def get_rules_description(self) -> List[Dict[str, Any]]:
        """Get a human-readable description of all active rules (alias for get_rules_summary)."""
        return self.get_rules_summary()
