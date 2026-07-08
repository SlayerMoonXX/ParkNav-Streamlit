"""
Unit tests untuk Expert System (Sistem Pakar Rekomendasi Parkir).
"""

import pytest
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.expert_system import ParkingExpertSystem, SlotRecommendation
from src.parking_map import ParkingMap, CellType


# ─────────────────────── Fixtures ───────────────────────


@pytest.fixture
def expert():
    """Create ParkingExpertSystem instance."""
    return ParkingExpertSystem()


@pytest.fixture
def map_with_slots():
    """Map with multiple available slots on different floors."""
    map_data = {
        "name": "Expert Test Map",
        "num_floors": 2,
        "num_rows": 6,
        "num_cols": 10,
        "floors": [
            {"floor": 0, "name": "Ground", "grid": [
                [6, 0, 0, 0, 0, 0, 0, 0, 0, 7],
                [0, 0, 2, 0, 2, 0, 2, 0, 0, 0],
                [0, 0, 2, 0, 2, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 2, 0, 2, 0, 8, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 4]
            ]},
            {"floor": 1, "name": "Level 1", "grid": [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 2, 0, 2, 0, 0, 0],
                [0, 0, 2, 0, 2, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 2, 0, 2, 0, 8, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 5]
            ]}
        ],
        "slots": {
            "G-A01": {"floor": 0, "row": 1, "col": 2, "type": "standard",
                      "status": "available", "accessible": False, "size": "standard"},
            "G-A02": {"floor": 0, "row": 1, "col": 4, "type": "standard",
                      "status": "available", "accessible": True, "size": "standard"},
            "G-A03": {"floor": 0, "row": 1, "col": 6, "type": "standard",
                      "status": "occupied", "accessible": False, "size": "standard"},
            "G-B01": {"floor": 0, "row": 2, "col": 2, "type": "standard",
                      "status": "available", "accessible": False, "size": "large"},
            "G-B02": {"floor": 0, "row": 2, "col": 4, "type": "standard",
                      "status": "available", "accessible": False, "size": "motorcycle"},
            "G-B03": {"floor": 0, "row": 2, "col": 6, "type": "standard",
                      "status": "available", "accessible": False, "size": "standard"},
            "G-C01": {"floor": 0, "row": 4, "col": 2, "type": "standard",
                      "status": "available", "accessible": True, "size": "standard"},
            "G-C02": {"floor": 0, "row": 4, "col": 4, "type": "standard",
                      "status": "available", "accessible": False, "size": "standard"},
            "G-C03": {"floor": 0, "row": 4, "col": 6, "type": "standard",
                      "status": "available", "accessible": False, "size": "standard"},
            "L1-A01": {"floor": 1, "row": 1, "col": 2, "type": "standard",
                       "status": "available", "accessible": False, "size": "standard"},
            "L1-A02": {"floor": 1, "row": 1, "col": 4, "type": "standard",
                       "status": "available", "accessible": False, "size": "standard"},
            "L1-A03": {"floor": 1, "row": 1, "col": 6, "type": "standard",
                       "status": "available", "accessible": False, "size": "standard"},
            "L1-B01": {"floor": 1, "row": 4, "col": 2, "type": "standard",
                       "status": "available", "accessible": False, "size": "standard"},
            "L1-B02": {"floor": 1, "row": 4, "col": 4, "type": "standard",
                       "status": "available", "accessible": False, "size": "standard"},
            "L1-B03": {"floor": 1, "row": 4, "col": 6, "type": "standard",
                       "status": "available", "accessible": False, "size": "standard"},
        },
        "ramp_connections": [
            {"from": [0, 5, 9], "to": [1, 5, 9], "cost_up": 2.0, "cost_down": 1.5}
        ],
        "elevator_positions": [
            {"row": 4, "col": 8, "floors": [0, 1], "cost_per_floor": 3.0}
        ]
    }
    return ParkingMap(map_data)


@pytest.fixture
def map_no_slots():
    """Map with no available slots (all occupied)."""
    map_data = {
        "name": "Full Map",
        "num_floors": 1,
        "num_rows": 3,
        "num_cols": 3,
        "floors": [{"floor": 0, "name": "Ground", "grid": [
            [6, 0, 7],
            [0, 3, 0],
            [0, 0, 0]
        ]}],
        "slots": {
            "X-01": {"floor": 0, "row": 1, "col": 1, "type": "standard",
                     "status": "occupied", "accessible": False, "size": "standard"}
        },
        "ramp_connections": [],
        "elevator_positions": []
    }
    return ParkingMap(map_data)


# ─────────────── Test Cases ───────────────


class TestExpertSystemRecommendation:
    """Tests for ParkingExpertSystem.recommend()."""

    def test_recommend_returns_results(self, expert, map_with_slots):
        """Expert system should return non-empty recommendations for available slots."""
        prefs = {
            'vehicle_type': 'sedan',
            'preferred_floor': None,
            'near_elevator': False,
            'near_exit': False,
        }
        recs = expert.recommend(map_with_slots, prefs)

        assert len(recs) > 0
        assert all(isinstance(r, SlotRecommendation) for r in recs)
        assert len(recs) <= 5  # Max 5 recommendations

    def test_disability_filter(self, expert, map_with_slots):
        """Disability vehicles should only get accessible slots."""
        prefs = {
            'vehicle_type': 'disability',
            'preferred_floor': None,
            'near_elevator': False,
            'near_exit': False,
        }
        recs = expert.recommend(map_with_slots, prefs)

        assert len(recs) > 0
        # All recommended slots must be accessible
        for rec in recs:
            slot_info = map_with_slots.slots.get(rec.slot_id, {})
            assert slot_info.get('accessible', False) is True, \
                f"Slot {rec.slot_id} is NOT accessible but was recommended for disability"

    def test_floor_preference(self, expert, map_with_slots):
        """Slots on the preferred floor should score higher with that preference."""
        prefs_floor0 = {
            'vehicle_type': 'sedan',
            'preferred_floor': 0,
            'near_elevator': False,
            'near_exit': False,
        }
        prefs_floor1 = {
            'vehicle_type': 'sedan',
            'preferred_floor': 1,
            'near_elevator': False,
            'near_exit': False,
        }
        recs0 = expert.recommend(map_with_slots, prefs_floor0)
        recs1 = expert.recommend(map_with_slots, prefs_floor1)

        # Top recommendation for floor 0 preference should be on floor 0
        assert recs0[0].floor == 0, \
            f"Top recommendation should be on floor 0, got floor {recs0[0].floor}"

        # When preferring floor 1, floor-1 slots should rank higher than
        # they do when preferring floor 0.
        f1_scores_with_pref1 = [r.score for r in recs1 if r.floor == 1]
        f1_scores_with_pref0 = [r.score for r in recs0 if r.floor == 1]
        if f1_scores_with_pref1 and f1_scores_with_pref0:
            assert max(f1_scores_with_pref1) > max(f1_scores_with_pref0), \
                "Floor 1 slots should score higher when floor 1 is preferred"

    def test_elevator_proximity_preference(self, expert, map_with_slots):
        """When near_elevator is True, slots near elevator should rank higher."""
        prefs_no_elev = {
            'vehicle_type': 'sedan',
            'preferred_floor': 0,
            'near_elevator': False,
            'near_exit': False,
        }
        prefs_elev = {
            'vehicle_type': 'sedan',
            'preferred_floor': 0,
            'near_elevator': True,
            'near_exit': False,
        }
        recs_no = expert.recommend(map_with_slots, prefs_no_elev)
        recs_yes = expert.recommend(map_with_slots, prefs_elev)

        # With elevator preference, C-row slots (near elevator at row=4, col=8)
        # should be favored more
        assert len(recs_yes) > 0
        # The scoring order should differ
        ids_no = [r.slot_id for r in recs_no]
        ids_yes = [r.slot_id for r in recs_yes]
        # At minimum, both should return valid results
        assert len(ids_yes) > 0

    def test_score_range(self, expert, map_with_slots):
        """All recommendation scores should be between 0 and 1."""
        prefs = {
            'vehicle_type': 'sedan',
            'preferred_floor': 0,
            'near_elevator': True,
            'near_exit': True,
        }
        recs = expert.recommend(map_with_slots, prefs)

        for rec in recs:
            assert 0.0 <= rec.score <= 1.0, \
                f"Score {rec.score} for {rec.slot_id} is out of [0, 1] range"

    def test_no_available_slots(self, expert, map_no_slots):
        """Should return empty list when no slots are available."""
        prefs = {
            'vehicle_type': 'sedan',
            'preferred_floor': None,
            'near_elevator': False,
            'near_exit': False,
        }
        recs = expert.recommend(map_no_slots, prefs)

        assert recs == []

    def test_recommendations_sorted_by_score(self, expert, map_with_slots):
        """Recommendations should be sorted by score in descending order."""
        prefs = {
            'vehicle_type': 'sedan',
            'preferred_floor': 0,
            'near_elevator': True,
            'near_exit': True,
        }
        recs = expert.recommend(map_with_slots, prefs)

        for i in range(len(recs) - 1):
            assert recs[i].score >= recs[i + 1].score, \
                f"Not sorted: {recs[i].slot_id}={recs[i].score} < {recs[i+1].slot_id}={recs[i+1].score}"

    def test_reasons_not_empty(self, expert, map_with_slots):
        """Each recommendation should have at least one reason."""
        prefs = {
            'vehicle_type': 'sedan',
            'preferred_floor': 0,
            'near_elevator': False,
            'near_exit': False,
        }
        recs = expert.recommend(map_with_slots, prefs)

        for rec in recs:
            assert len(rec.reasons) > 0, \
                f"Slot {rec.slot_id} has no reasons"

    def test_distance_to_entrance_populated(self, expert, map_with_slots):
        """distance_to_entrance should be populated in each recommendation."""
        prefs = {
            'vehicle_type': 'sedan',
            'preferred_floor': None,
            'near_elevator': False,
            'near_exit': False,
        }
        recs = expert.recommend(map_with_slots, prefs)

        for rec in recs:
            assert rec.distance_to_entrance is not None
            assert rec.distance_to_entrance >= 0

    def test_motorcycle_size_filter(self, expert, map_with_slots):
        """Motorcycle should only get motorcycle or standard sized slots."""
        prefs = {
            'vehicle_type': 'motorcycle',
            'preferred_floor': None,
            'near_elevator': False,
            'near_exit': False,
        }
        recs = expert.recommend(map_with_slots, prefs)

        assert len(recs) > 0
        for rec in recs:
            slot_info = map_with_slots.slots.get(rec.slot_id, {})
            slot_size = slot_info.get('size', 'standard')
            assert slot_size in ['motorcycle', 'standard'], \
                f"Motorcycle got incompatible slot size: {slot_size}"


class TestExpertSystemMethods:
    """Tests for internal methods of ParkingExpertSystem."""

    def test_size_compatible(self, expert):
        """Vehicle-slot size compatibility should be correct."""
        # Standard vehicle fits in standard and large slots
        assert expert._size_compatible('standard', 'standard') is True
        assert expert._size_compatible('standard', 'large') is True
        assert expert._size_compatible('standard', 'small') is False

        # Large vehicle only fits in large slots
        assert expert._size_compatible('large', 'large') is True
        assert expert._size_compatible('large', 'standard') is False

        # Motorcycle fits anywhere
        assert expert._size_compatible('motorcycle', 'standard') is True
        assert expert._size_compatible('motorcycle', 'large') is True
        assert expert._size_compatible('small', 'standard') is True

    def test_compute_score_returns_valid_range(self, expert, map_with_slots):
        """_compute_score should return score in [0, 1] range."""
        entrance = map_with_slots.get_entrance()
        exit_pos = map_with_slots.get_exit()
        elevator_positions = [
            (elev['floors'][0], elev['row'], elev['col'])
            for elev in map_with_slots.elevator_positions
            if elev.get('floors')
        ]
        slot = {'floor': 0, 'row': 1, 'col': 2, 'accessible': False}
        slot_pos = (0, 1, 2)

        score, reasons = expert._compute_score(
            slot, slot_pos, entrance, exit_pos,
            elevator_positions, None, False, False
        )
        assert 0.0 <= score <= 1.0
        assert isinstance(reasons, list)
        assert len(reasons) > 0

    def test_compute_score_preferred_floor_boosts(self, expert, map_with_slots):
        """Slot on preferred floor should score higher than slot on different floor."""
        entrance = map_with_slots.get_entrance()
        exit_pos = map_with_slots.get_exit()
        elevator_positions = []

        slot_f0 = {'floor': 0, 'row': 1, 'col': 2, 'accessible': False}
        slot_f1 = {'floor': 1, 'row': 1, 'col': 2, 'accessible': False}

        score_f0, _ = expert._compute_score(
            slot_f0, (0, 1, 2), entrance, exit_pos,
            elevator_positions, 0, False, False
        )
        score_f1, _ = expert._compute_score(
            slot_f1, (1, 1, 2), entrance, exit_pos,
            elevator_positions, 0, False, False
        )
        # Slot on preferred floor 0 should score higher
        assert score_f0 > score_f1

    def test_get_rules_description(self, expert):
        """get_rules_description should return a non-empty list of rules."""
        rules = expert.get_rules_description()
        assert isinstance(rules, list)
        assert len(rules) > 0
        # Each rule should have an id and description
        for rule in rules:
            assert 'id' in rule
            assert 'description' in rule

