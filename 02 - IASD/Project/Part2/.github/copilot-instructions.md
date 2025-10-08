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


