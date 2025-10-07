# Copilot Instructions: Autonomous Mars Gardening Robot

## Project Overview
This is an AI search problem implementation for an autonomous gardening robot on Mars. The robot must navigate a grid, avoid obstacles, refill water at origin (0,0), and water plants within deadlines using **uninformed search** (breadth-first graph search or uniform cost search).

## Architecture & Core Components

### State Representation
States are tuples: `(robot_pos, water_level, time, watered_plants)`
- `robot_pos`: (x,y) coordinates where x=col, y=row  
- `water_level`: current water amount
- `time`: elapsed time units
- `watered_plants`: bitmask of watered plants (0-indexed)

### Key Files
- **`solution.py`**: Main problem implementation with `GardenerProblem` class
- **`search.py`**: Generic search algorithms (use `breadth_first_graph_search` or `uniform_cost_search`)
- **`utils.py`**: Supporting utilities and data structures
- **`test_debug.py`**: Debug version without optimizations
- **`public2/*.dat`**: Test cases with expected `.plan` solutions

## Critical Semantics & Constraints

### Watering Deadline Semantics
- **Watering completes at t+1**: From time `t`, watering finishes at `t+1`
- **Legal watering**: Requires `t+1 ≤ dk` (not `t ≤ dk` or `t < dk`)
- Must be consistent in both `actions()` and `result()` methods

### Water Refill Rules
- **Refill trigger**: Only when arriving at (0,0) via a move action
- **Never during**: Watering action or while idling at origin
- **Refill amount**: Always refills to full capacity `W0`

### Goal Test
- **Completion**: All plant cells watered (bitmask full: `watered_plants == (1 << P) - 1`)
- **No return required**: Robot doesn't need to return to origin

### Input Constraints
- **Origin constraint**: Position (0,0) must be empty (value 0)
- **Grid values**: Only -1 (obstacle), 0 (empty), or 1..K (plant types)
- **Plant types**: Read exactly K = max(grid) plant-type definitions

## Critical Patterns & Conventions

### Data File Format
```
N M W0          # Grid height, width, water capacity
[N rows of grid data]  # 0=empty, -1=obstacle, 1..K=plant types
[K plant definitions]  # "wk dk" (water needed, deadline)
```

### Plant Management System
```python
# Plants indexed by grid position for O(1) lookup
self.plants = {}  # (x,y) -> (plant_type, wk, dk)
self.plant_idx = {}  # (x,y) -> bitmask_index

# Bitmask operations for watered state
watered_plants |= (1 << plant_index)  # Mark watered
((watered_plants >> i) & 1) == 0      # Check not watered
```

## Exact Distances Precompute (Uninformed Optimization)

### BFS Distance Maps
Precompute exact grid distances using BFS for feasibility pruning:

```python
# Distance from origin to all positions
dist_to_origin[(x,y)] = BFS_distance_from_origin_to_(x,y)

# Distance from each plant position to all positions  
dist_to_plant[i][(x,y)] = BFS_distance_from_plant_i_to_(x,y)
```

### Implementation
- Run BFS once from (0,0) to build `dist_to_origin`
- For each plant `i` at position `pi`, run BFS from `pi` to build `dist_to_plant[i]`
- These are exact unit-cost distances on the grid (accounting for obstacles)

## Feasibility Pruning (Exact, Uninformed)

### 1. Global Missed-Deadline Cut
Before generating actions, check if ALL unwatered plants have missed deadlines:
```python
for all unwatered plants i:
    finish_time = time + dist_to_plant[i][pos] + 1
    if finish_time > dk[i]:
        all_missed = True
if all_missed: return []  # Dead end
```

### 2. Must-Refill Cut  
When `water < min(wk)` of remaining plants and `pos != (0,0)`:
```python
d0 = dist_to_origin[pos]
for all remaining plants i:
    finish_time = time + d0 + dist_to_plant[i][(0,0)] + 1  
    if finish_time > dk[i]:
        all_missed_after_refill = True
if all_missed_after_refill: return []
```

## Performance Optimizations

### Pareto Frontier Pruning
Keep only non-dominated (time, water) pairs per (position, watered_mask):
```python
# Prune if exists (t*, w*) with t* ≤ time and w* ≥ water
frontier = visited_states[(pos, mask)]
for (t, w) in frontier:
    if t <= time and w >= water:
        return True  # Dominated -> prune
```

### State Management
- **Reset pruning state**: Clear `visited_states` at start of each `solve()`
- **Deterministic plant indexing**: Row-major order ensures stable bitmask indices
- **Early unsatisfiability**: Fail fast if any plant requires `wk > W0`

## Action Generation & Ordering

### Action Priorities
1. **Water first**: Emit 'W' action first when legal (helps solutions surface earlier)
2. **Movement**: Then emit L,R,U,D moves

### Movement Validation
```python
# Bounds check: 0 ≤ nx < width and 0 ≤ ny < height
# Obstacle check: new_pos not in self.obstacles
```

### Watering Constraints
```python
# Legal watering at time t requires:
# 1. Plant exists at position
# 2. Plant not already watered  
# 3. Sufficient water: water_level >= wk[i]
# 4. Within deadline: t + 1 <= dk[i]
```

## Development Workflow

### Running Tests
```bash
python solution.py  # Runs configured example
python test_debug.py  # Runs without optimizations
```

### Search Algorithm Choice
Use either:
- `breadth_first_graph_search(self)` - Standard BFS
- `uniform_cost_search(self)` - Often better with AIMA framework

Both are uninformed; pick one and stay consistent.

### Key Debugging Points
- Monitor `self.visited_states` size for pruning effectiveness
- Verify plant indexing is deterministic (row-major order)
- Check exact distance precomputation accuracy
- Validate deadline semantics: `t+1 ≤ dk`

## Common Pitfalls
- **Deadline semantics**: Must use `t+1 ≤ dk`, not `t ≤ dk` or `t < dk`
- **Refill timing**: Only after move into origin, never during watering
- **Coordinate system**: Grid uses (row, col) but positions are (x=col, y=row)
- **Reset state**: Clear pruning data structures between solves
- **Exact vs heuristic**: All optimizations must be exact/sound, no distance-based heuristics