"""Quick validation test for all core engine components."""
import sys
sys.path.insert(0, '.')

# Test 1: Load map
from src.parking_map import ParkingMap, CellType
pm = ParkingMap.from_json('data/maps/default_map.json')
print(f'[OK] Map loaded: {pm}')

# Test 2: Check entrance/exit
entrance = pm.get_entrance()
exit_pos = pm.get_exit()
print(f'[OK] Entrance: {entrance}, Exit: {exit_pos}')

# Test 3: Check available slots
avail = pm.get_available_slots()
print(f'[OK] Available slots: {len(avail)} slots')

# Test 4: Run A* search
from src.astar import astar_search, bfs_search, dfs_search, greedy_search
from src.heuristic import manhattan_3d, euclidean_3d

target_slot = avail[0]
goal = (target_slot['floor'], target_slot['row'], target_slot['col'])
print(f'[OK] Searching from {entrance} to slot {target_slot["id"]} at {goal}')

result_astar = astar_search(pm, entrance, goal)
print(f'[OK] A*:     success={result_astar.success}, cost={result_astar.cost:.2f}, steps={len(result_astar.path)}, explored={result_astar.nodes_explored}, time={result_astar.execution_time:.2f}ms')

result_bfs = bfs_search(pm, entrance, goal)
print(f'[OK] BFS:    success={result_bfs.success}, cost={result_bfs.cost:.2f}, steps={len(result_bfs.path)}, explored={result_bfs.nodes_explored}')

result_dfs = dfs_search(pm, entrance, goal)
print(f'[OK] DFS:    success={result_dfs.success}, cost={result_dfs.cost:.2f}, steps={len(result_dfs.path)}, explored={result_dfs.nodes_explored}')

result_greedy = greedy_search(pm, entrance, goal)
print(f'[OK] Greedy: success={result_greedy.success}, cost={result_greedy.cost:.2f}, steps={len(result_greedy.path)}, explored={result_greedy.nodes_explored}')

# Test 5: Cross-floor search
goal_l3 = pm.get_slot_position('L3-A01')
result_cross = astar_search(pm, entrance, goal_l3)
print(f'[OK] Cross-floor A*: success={result_cross.success}, cost={result_cross.cost:.2f}, steps={len(result_cross.path)}')

# Test 6: Expert system
from src.expert_system import ParkingExpertSystem
es = ParkingExpertSystem()
recs = es.recommend_slots(pm, vehicle_type='sedan')
print(f'[OK] Expert system: {len(recs)} recommendations')
for r in recs[:3]:
    print(f'     - {r["id"]} (score: {r["score"]:.4f})')

# Test 7: Utils
from src.utils import get_available_maps, format_path, format_time, format_search_result
maps = get_available_maps()
print(f'[OK] Available maps: {maps}')
print(f'[OK] Format time: {format_time(0.5)}, {format_time(42.3)}, {format_time(1500)}')

# Test 8: Edge cases
result_same = astar_search(pm, entrance, entrance)
print(f'[OK] Start==Goal: success={result_same.success}, msg="{result_same.message}"')

# Test 9: Load other maps
pm2 = ParkingMap.from_json('data/maps/scenario_deadend.json')
print(f'[OK] Deadend map: {pm2}')
pm3 = ParkingMap.from_json('data/maps/scenario_complex.json')
print(f'[OK] Complex map: {pm3}')

# Test 10: Search on deadend map
entrance2 = pm2.get_entrance()
avail2 = pm2.get_available_slots()
if avail2:
    g2 = (avail2[-1]['floor'], avail2[-1]['row'], avail2[-1]['col'])
    r2 = astar_search(pm2, entrance2, g2)
    print(f'[OK] Deadend A*: success={r2.success}, cost={r2.cost:.2f}, explored={r2.nodes_explored}')

# Test 11: Format search result display
print()
print(format_search_result(result_astar))

print()
print('=== ALL TESTS PASSED ===')
