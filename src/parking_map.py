"""Data model for multi-story parking structures.

This module defines the ParkingMap class which represents a multi-floor parking
structure with support for roads, walls, parking slots, ramps, elevators,
entrances, and exits.
"""

import json
from typing import List, Tuple, Optional, Dict, Any


class CellType:
    """Constants representing cell types in the parking grid."""
    ROAD = 0
    WALL = 1
    SLOT_AVAILABLE = 2
    SLOT_OCCUPIED = 3
    RAMP_UP = 4
    RAMP_DOWN = 5
    ENTRANCE = 6
    EXIT = 7
    ELEVATOR = 8

    NAMES = {
        0: 'Road', 1: 'Wall', 2: 'Available Slot', 3: 'Occupied Slot',
        4: 'Ramp Up', 5: 'Ramp Down', 6: 'Entrance', 7: 'Exit', 8: 'Elevator'
    }
    COLORS = {
        0: '#E8E8E8', 1: '#2C2C2C', 2: '#4CAF50', 3: '#F44336',
        4: '#FF9800', 5: '#FF5722', 6: '#2196F3', 7: '#00BCD4', 8: '#9C27B0'
    }

    WALKABLE = {ROAD, SLOT_AVAILABLE, RAMP_UP, RAMP_DOWN, ENTRANCE, EXIT, ELEVATOR}


class ParkingMap:
    """Represents a multi-floor parking structure with navigation support.

    Attributes:
        name: Name of the parking map.
        num_floors: Number of floors in the structure.
        num_rows: Number of rows per floor grid.
        num_cols: Number of columns per floor grid.
        floors: List of 2D grids (list of lists of ints) per floor.
        floor_names: Display names for each floor.
        slots: Dictionary of slot_id -> slot info dict.
        ramp_connections: List of ramp connection definitions.
        elevator_positions: List of elevator position definitions.
    """

    def __init__(self, map_data: dict):
        """Initialize ParkingMap from parsed JSON map data.

        Args:
            map_data: Dictionary containing the full map specification.

        Raises:
            KeyError: If required fields are missing from map_data.
            ValueError: If grid dimensions don't match specification.
        """
        try:
            self.name = map_data['name']
            self.num_floors = map_data['num_floors']
            self.num_rows = map_data['num_rows']
            self.num_cols = map_data['num_cols']
            self.floors = [floor_data['grid'] for floor_data in map_data['floors']]
            self.floor_names = [
                floor_data.get('name', f'Floor {i + 1}')
                for i, floor_data in enumerate(map_data['floors'])
            ]
            self.slots = map_data.get('slots', {})
            self.ramp_connections = map_data.get('ramp_connections', [])
            self.elevator_positions = map_data.get('elevator_positions', [])
        except KeyError as e:
            raise KeyError(f"Missing required field in map data: {e}")

        # Validate grid dimensions
        for f_idx, grid in enumerate(self.floors):
            if len(grid) != self.num_rows:
                raise ValueError(
                    f"Floor {f_idx} has {len(grid)} rows, expected {self.num_rows}"
                )
            for r_idx, row in enumerate(grid):
                if len(row) != self.num_cols:
                    raise ValueError(
                        f"Floor {f_idx}, row {r_idx} has {len(row)} cols, "
                        f"expected {self.num_cols}"
                    )

        self._connections: Dict[Tuple[int, int, int], List[Tuple[int, int, int, float]]] = {}
        self._build_connection_map()

    def _build_connection_map(self):
        """Build lookup dict for inter-floor connections (ramps and elevators)."""
        self._connections = {}

        # Ramp connections (bidirectional)
        for ramp in self.ramp_connections:
            from_pos = tuple(ramp['from'])  # (floor, row, col)
            to_pos = tuple(ramp['to'])      # (floor, row, col)
            cost_up = ramp.get('cost_up', 2.0)
            cost_down = ramp.get('cost_down', 1.5)

            # Going up: from -> to
            from_key = (from_pos[0], from_pos[1], from_pos[2])
            to_entry = (to_pos[0], to_pos[1], to_pos[2], cost_up)
            self._connections.setdefault(from_key, []).append(to_entry)

            # Going down: to -> from
            to_key = (to_pos[0], to_pos[1], to_pos[2])
            from_entry = (from_pos[0], from_pos[1], from_pos[2], cost_down)
            self._connections.setdefault(to_key, []).append(from_entry)

        # Elevator connections (all pairs of floors served by each elevator)
        for elevator in self.elevator_positions:
            elev_row = elevator['row']
            elev_col = elevator['col']
            served_floors = elevator['floors']
            cost_per_floor = elevator.get('cost_per_floor', 1.0)

            for i, floor_a in enumerate(served_floors):
                for floor_b in served_floors[i + 1:]:
                    floor_diff = abs(floor_b - floor_a)
                    cost = cost_per_floor * floor_diff

                    key_a = (floor_a, elev_row, elev_col)
                    key_b = (floor_b, elev_row, elev_col)

                    self._connections.setdefault(key_a, []).append(
                        (floor_b, elev_row, elev_col, cost)
                    )
                    self._connections.setdefault(key_b, []).append(
                        (floor_a, elev_row, elev_col, cost)
                    )

    def get_cell(self, floor: int, row: int, col: int) -> int:
        """Get the cell type value at the given position.

        Args:
            floor: Floor index (0-based).
            row: Row index.
            col: Column index.

        Returns:
            Integer cell type value.

        Raises:
            IndexError: If position is out of bounds.
        """
        if not self.is_valid(floor, row, col):
            raise IndexError(f"Position ({floor}, {row}, {col}) is out of bounds")
        return self.floors[floor][row][col]

    def set_cell(self, floor: int, row: int, col: int, value: int):
        """Set the cell type value at the given position.

        Args:
            floor: Floor index (0-based).
            row: Row index.
            col: Column index.
            value: Cell type value to set.

        Raises:
            IndexError: If position is out of bounds.
        """
        if not self.is_valid(floor, row, col):
            raise IndexError(f"Position ({floor}, {row}, {col}) is out of bounds")
        self.floors[floor][row][col] = value

    def is_valid(self, floor: int, row: int, col: int) -> bool:
        """Check if a position is within the grid boundaries."""
        return (
            0 <= floor < self.num_floors
            and 0 <= row < self.num_rows
            and 0 <= col < self.num_cols
        )

    def is_walkable(self, floor: int, row: int, col: int) -> bool:
        """Check if a cell can be traversed.

        Walkable cells: ROAD, SLOT_AVAILABLE, RAMP_UP, RAMP_DOWN, ENTRANCE, EXIT, ELEVATOR.
        Non-walkable: WALL, SLOT_OCCUPIED.
        """
        if not self.is_valid(floor, row, col):
            return False
        return self.floors[floor][row][col] in CellType.WALKABLE

    def get_neighbors(self, floor: int, row: int, col: int) -> List[Tuple[int, int, int, float]]:
        """Get reachable neighboring positions with movement costs.

        Args:
            floor: Current floor index.
            row: Current row index.
            col: Current column index.

        Returns:
            List of (floor, row, col, cost) tuples for valid neighbors.
        """
        neighbors = []

        # Same-floor 4-directional movement (cost = 1.0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_valid(floor, nr, nc) and self.is_walkable(floor, nr, nc):
                neighbors.append((floor, nr, nc, 1.0))

        # Inter-floor connections from pre-built connection map
        key = (floor, row, col)
        if key in self._connections:
            for target_floor, target_row, target_col, cost in self._connections[key]:
                if self.is_valid(target_floor, target_row, target_col):
                    neighbors.append((target_floor, target_row, target_col, cost))

        return neighbors

    def get_available_slots(self) -> List[Dict[str, Any]]:
        """Get all parking slots with status 'available'.

        Returns:
            List of slot info dicts with the slot_id added as 'id' key.
        """
        available = []
        for slot_id, slot_info in self.slots.items():
            if slot_info.get('status') == 'available':
                slot_copy = dict(slot_info)
                slot_copy['id'] = slot_id
                available.append(slot_copy)
        return available

    def get_entrance(self) -> Optional[Tuple[int, int, int]]:
        """Find the ENTRANCE cell position.

        Returns:
            (floor, row, col) tuple or None if no entrance found.
        """
        for f in range(self.num_floors):
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if self.floors[f][r][c] == CellType.ENTRANCE:
                        return (f, r, c)
        return None

    def get_exit(self) -> Optional[Tuple[int, int, int]]:
        """Find the EXIT cell position.

        Returns:
            (floor, row, col) tuple or None if no exit found.
        """
        for f in range(self.num_floors):
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if self.floors[f][r][c] == CellType.EXIT:
                        return (f, r, c)
        return None

    def get_slot_position(self, slot_id: str) -> Optional[Tuple[int, int, int]]:
        """Get the grid position of a parking slot by its ID.

        Args:
            slot_id: The slot identifier (e.g., 'L1-A01').

        Returns:
            (floor, row, col) tuple or None if slot not found.
        """
        if slot_id not in self.slots:
            return None
        slot = self.slots[slot_id]
        return (slot['floor'], slot['row'], slot['col'])

    def toggle_slot(self, slot_id: str):
        """Toggle a parking slot between available and occupied.

        Args:
            slot_id: The slot identifier to toggle.

        Raises:
            KeyError: If slot_id is not found.
        """
        if slot_id not in self.slots:
            raise KeyError(f"Slot '{slot_id}' not found")

        slot = self.slots[slot_id]
        floor, row, col = slot['floor'], slot['row'], slot['col']

        if slot['status'] == 'available':
            slot['status'] = 'occupied'
            self.set_cell(floor, row, col, CellType.SLOT_OCCUPIED)
        else:
            slot['status'] = 'available'
            self.set_cell(floor, row, col, CellType.SLOT_AVAILABLE)

    def to_dict(self) -> dict:
        """Serialize the parking map back to a dictionary."""
        return {
            'name': self.name,
            'num_floors': self.num_floors,
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'floors': [
                {
                    'floor': i,
                    'name': self.floor_names[i],
                    'grid': self.floors[i]
                }
                for i in range(self.num_floors)
            ],
            'slots': self.slots,
            'ramp_connections': self.ramp_connections,
            'elevator_positions': self.elevator_positions,
        }

    @classmethod
    def from_json(cls, filepath: str) -> 'ParkingMap':
        """Load a ParkingMap from a JSON file.

        Args:
            filepath: Path to the JSON map file.

        Returns:
            ParkingMap instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data)

    def __repr__(self) -> str:
        return (
            f"ParkingMap(name='{self.name}', floors={self.num_floors}, "
            f"size={self.num_rows}x{self.num_cols}, slots={len(self.slots)})"
        )
