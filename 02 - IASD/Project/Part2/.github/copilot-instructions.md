# Copilot Instructions: Autonomous Mars Gardening Robot

## Project Overview
This is an AI search problem implementation for an autonomous gardening robot on Mars. The robot must navigate a grid, avoid obstacles, refill water at origin (0,0), and water plants within deadlines using **uninformed search** (breadth-first graph search or uniform cost search).

*No HEURISTICS or A* allowed!* Only uninformed search.


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
- **`env/`**: Python virtual environment with numpy dependency

### Critical Implementation Details
- **Coordinate System**: Grid uses (x,y) where x=column, y=row. Map access: `self.map[y][x]`
- **Plant Indexing**: Row-major order for deterministic bitmask indexes: `for r in range(N): for c in range(M)`
- **Performance**: Uses Pareto frontier pruning and deadline-based early termination

## Development Workflow

### Running Tests
```bash
# Activate environment first
source env/bin/activate

# Run all test cases ex0-ex9 (takes ~40s total, ex7-ex8 are slow)
python solution.py

# Test specific example
python -c "
from solution import GardenerProblem
with open('public2/ex0.dat') as f:
    p = GardenerProblem()
    p.load(f)
    print(p.solve())
"
```

### Performance Characteristics  
- **Fast examples** (ex0-ex6, ex9): < 1 second each
- **Slow examples** (ex7, ex8): 17-20 seconds due to large search space
- **Total test suite**: ~40 seconds for all 10 examples

### Performance Optimization Patterns
- **Pareto Frontier Pruning**: `prune()` method maintains dominance frontier over (time, water) per (position, watered_plants)
- **Deadline Pruning**: Early termination when any unwatered plant deadline is exceeded  
- **Action Ordering**: Water action emitted first when legal (helps solutions surface earlier)
- **State Pruning**: Simplified state without backtracking prevention for better performance

## Critical Semantics & Constraints

### Watering Deadline Semantics
- **Watering completes at t+1**: From time `t`, watering finishes at `t+1`
- **Legal watering**: Requires `t+1 ≤ dk` (not `t ≤ dk` or `t < dk`)
- **Implementation**: Must be consistent in both `actions()` and `result()` methods

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

## Test Cases & Expected Solutions

### Example 0 (ex0) - Simple 3x4 Grid
**Input:** `public2/ex0.dat`
```
# This is N, M, and W0
3 4 100

# The map
0  0 1  0
0 -1 2 -1
1  0 1 -1

# Two plant types
2 10
3 15
```
**Expected Solution:** `DDWRRWUWUW` (10 actions)

---

### Example 1 (ex1) - 5x11 Grid with Water Refill
**Input:** `public2/ex1.dat`
```
5 11 6

0 0 1 0 0 0 0 0 0 0 0
0 0 1 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 2
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0

1 100
6 12
```
**Expected Solution:** `RRRRRRRRRRDDWUULLLLLLLLLLRRWDWLDW` (33 actions)

---

### Example 7 (ex7) - Large 10x10 Grid (SLOW - ~17-20s)
**Input:** `public2/ex7.dat`
```
10 10 200

0 0 0 0 0 0 0 0 2 2
0 0 0 0 0 0 0 0 0 2
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
3 0 0 0 0 0 0 0 0 1
3 3 0 0 0 0 0 0 1 1

2 100
5 100
1 100
```
**Expected Solution:** `DDDDDDDDWDWRWRRRRRRRWRWUWUUUUUUUWUWLW` (37 actions)
- Grid size: 10x10
- Water capacity: 200
- Plant types: 3 (type 1: 1 water/100 deadline, type 2: 2 water/100 deadline, type 3: 5 water/100 deadline)
- Total plants: 12
- Performance note: Large search space due to many plants and long deadlines

---

### Example 8 (ex8) - Large 10x10 Grid with Low Water (SLOW - ~17-20s)
**Input:** `public2/ex8.dat`
```
10 10 10

0 0 0 0 0 0 0 2 2 2
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 1
3 3 0 0 0 0 0 0 1 1

1 100
2 100
3 100
```
**Expected Solution:** `RRRRRRRWRWRWLLLLLLLLLDDDDDDDDDWRWRRRRRRRWRWUW` (44 actions)
- Grid size: 10x10
- Water capacity: **10** (very limited - requires multiple refills at origin)
- Plant types: 3 (type 1: 1 water/100 deadline, type 2: 2 water/100 deadline, type 3: 3 water/100 deadline)
- Total plants: 11
- Performance note: Complex due to low water capacity requiring strategic refill planning

---

### Example 9 (ex9) - 10x10 Grid with Tight Deadline
**Input:** `public2/ex9.dat`
```
10 10 100

0 0 0 0 0 0 0 0 2 2
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
3 3 0 0 0 0 0 0 1 1
3 3 0 0 0 0 0 0 1 1

1 22
2 100
3 100
```
**Expected Solution:** `RRRRRRRRDDDDDDDDWRWDWLWLLLLLLLWLWUWRWUUUUUUUURRRRRRRWRW` (52 actions)
- Grid size: 10x10
- Water capacity: 100
- Plant types: 3 (type 1: 1 water/**22 deadline** - TIGHT!, type 2: 2 water/100 deadline, type 3: 3 water/100 deadline)
- Total plants: 14
- Performance note: Tight deadline on type 1 plants requires careful ordering

---

### Test Case Summary
- **Fast examples** (ex0-ex6, ex9): Complete in < 1 second each
- **Slow examples** (ex7, ex8): Take 17-20 seconds due to:
  - Large 10x10 grids
  - Many plants (11-12 plants)
  - Long deadlines (100) = huge search space
  - ex8 additionally has low water capacity (10) requiring complex refill strategy
- **Total test suite runtime**: ~40 seconds for all 10 examples


