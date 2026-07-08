from src.utils import load_map
from src.astar import astar_search, bfs_search, dfs_search, greedy_search
from src.expert_system import ParkingExpertSystem

pm = load_map('default_map')
entrance = pm.get_entrance()
print(f"Map: {pm.name}, Floors: {pm.num_floors}, Total slots: {len(pm.slots)}")
print(f"Available: {len(pm.get_available_slots())}")
print(f"Entrance: {entrance}")
print()

# Find reachable slots on different floors
for target_floor in [0, 1, 2]:
    avail = pm.get_available_slots()
    for s in avail:
        if s['floor'] == target_floor:
            goal = (s['floor'], s['row'], s['col'])
            r = astar_search(pm, entrance, goal)
            if r.success:
                sid = s['id']
                print(f"--- Floor {target_floor} target: {sid} at {goal} ---")
                a = astar_search(pm, entrance, goal)
                b = bfs_search(pm, entrance, goal)
                d = dfs_search(pm, entrance, goal)
                g = greedy_search(pm, entrance, goal)
                print(f"A*:     cost={a.cost:6.1f}, explored={a.nodes_explored:4d}, time={a.execution_time:.3f}ms, steps={len(a.path)}")
                print(f"BFS:    cost={b.cost:6.1f}, explored={b.nodes_explored:4d}, time={b.execution_time:.3f}ms, steps={len(b.path)}")
                print(f"DFS:    cost={d.cost:6.1f}, explored={d.nodes_explored:4d}, time={d.execution_time:.3f}ms, steps={len(d.path)}")
                print(f"Greedy: cost={g.cost:6.1f}, explored={g.nodes_explored:4d}, time={g.execution_time:.3f}ms, steps={len(g.path)}")
                print()
                break

# Dead-end scenario
print("=== Dead-End Scenario ===")
pm2 = load_map('scenario_deadend')
ent2 = pm2.get_entrance()
avail2 = pm2.get_available_slots()
for s2 in avail2:
    goal2 = (s2['floor'], s2['row'], s2['col'])
    r2 = astar_search(pm2, ent2, goal2)
    if r2.success and r2.nodes_explored > 10:
        print(f"Slot: {s2['id']}, explored={r2.nodes_explored}, cost={r2.cost:.1f}, steps={len(r2.path)}")
        break

# Expert system
print()
print("=== Expert System Recommendations ===")
expert = ParkingExpertSystem()
recs = expert.recommend(pm, {
    'vehicle_type': 'sedan',
    'preferred_floor': 0,
    'near_elevator': False,
    'near_exit': True
})
for i, r in enumerate(recs[:5]):
    print(f"#{i+1}: {r.slot_id} (Floor {r.floor+1}) score={r.score:.4f} dist={r.distance_to_entrance}")
    for reason in r.reasons:
        print(f"    -> {reason}")
