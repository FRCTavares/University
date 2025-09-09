# IASD Assignment #1 - Lean Project Plan

**Team:** Francisco, Marta and Diogo  
**Start Date:** September 9, 2025  
**Deadline:** September 26, 2025  
**Deliverable:** `solution.py` implementing the `GardenerProblem` class

---

# Project Summary

We are implementing an AI system for an autonomous gardening robot on Mars that must efficiently navigate a grid to water plants before their deadlines while managing limited water resources. This is essentially a **path planning and constraint satisfaction problem** where the robot moves in 4 directions (U/D/L/R) or waters plants (W), starting from position (0,0) which serves as the water refill station.

The core deliverable is a `GardenerProblem` class with two critical methods: `load(fh)` to parse input files containing grid layouts and plant specifications, and `check_solution(plan, verbose=False)` to validate action sequences against all constraints (boundaries, obstacles, water capacity, plant deadlines). We have 10 official test cases (ex0.dat through ex9.dat with corresponding .plan files) that our solution must pass.

Our approach focuses on clean, working code that handles the core requirements and edge cases without overengineering. The emphasis is on getting a robust solution that passes all official tests and handles basic error scenarios.

# Lean Roadmap (Sep 9 → Sep 26)

## Week 1: Core Implementation (Sep 9-15)

- **Sep 9-10**: Project setup, analyze official test files, design basic architecture
- **Sep 11-13**: Implement `load()` method and `check_solution()` method in parallel
- **Sep 14-15**: Basic integration testing, fix initial bugs

## Week 2: Testing & Validation (Sep 16-22)

- **Sep 16-18**: Test against all 10 official test cases (ex0-ex9), create basic edge case tests
- **Sep 19-20**: Bug fixes and refinements based on test results
- **Sep 21-22**: Code review, cleanup, and integration verification

## Week 3: Final Polish & Submission (Sep 23-26)

- **Sep 23-24**: Final testing round, handle any remaining edge cases
- **Sep 25**: Complete code review, prepare submission package
- **Sep 26**: Final validation and submit `solution.py` to Moodle

# Task Distribution

## Francisco: Input Parsing & Data Structures

**Primary Focus**: Implement `load(fh)` method and internal data representation

**Key Tasks**:

- Parse input file format (grid dimensions, grid data, plant specifications)
- Create robust data structures for grid, plant types, and robot state
- Handle file I/O with basic error checking
- Test parsing with all 10 official .dat files
- Integrate with Marta's validation logic

**Week 1**: Complete `load()` method with basic error handling
**Week 2**: Refine parsing based on integration testing
**Week 3**: Final validation and documentation

## Marta: Solution Validation & Robot Simulation

**Primary Focus**: Implement `check_solution(plan, verbose=False)` method

**Key Tasks**:

- Build robot simulation engine (position tracking, water management, time steps)
- Implement action execution (U/D/L/R moves and W watering)
- Validate all constraints (boundaries, obstacles, deadlines, water capacity)
- Provide clear error messages for invalid plans
- Test validation with all 10 official .plan files

**Week 1**: Complete core validation logic and robot simulation
**Week 2**: Refine constraint checking based on test results
**Week 3**: Final error handling and edge case management

## Diogo: Testing & Quality Assurance

**Primary Focus**: Ensure solution works correctly with official test cases

**Key Tasks**:

- Create automated test framework using official ex0-ex9 test pairs
- Design and implement basic edge case tests (empty grids, invalid actions, etc.)
- Coordinate integration testing between Francisco and Marta's components
- Track and document any failing test cases
- Prepare final submission validation

**Week 1**: Set up testing framework and run initial tests
**Week 2**: Comprehensive testing with all official cases, create edge case tests
**Week 3**: Final validation and submission preparation

## Joint Responsibilities (All Team Members)

- **Architecture decisions**: Collaborative design of class interfaces and data flow
- **Code reviews**: Each major component reviewed by other team members  
- **Integration debugging**: Joint sessions to resolve interface issues
- **Final submission**: All members verify final `solution.py` before submission
- **Emergency support**: Help other team members if they encounter blockers

**Key Coordination Points**:

- Daily brief check-ins via team chat
- Wednesday mid-week integration sessions
- Friday end-of-week progress reviews
- Final submission preparation on Sep 25-26

---

# Technical Implementation Details

## Required Class Structure

```python
import search

class GardenerProblem(search.Problem):
    def __init__(self):
        self.grid = None           # 2D array: -1=obstacle, 0=empty, k=plant_type
        self.plant_types = {}      # {plant_type: (water_needed, deadline)}
        self.water_capacity = 0    # W0 from input
        self.grid_size = (0, 0)    # (N, M) dimensions
        self.plant_positions = {}  # {plant_type: [(x,y), ...]} for quick lookup
    
    def load(self, fh):
        """Parse input file and populate data structures"""
        # Francisco's implementation
        pass
    
    def check_solution(self, plan, verbose=False):
        """Validate action sequence against all constraints"""
        # Marta's implementation
        pass
```

## Critical Implementation Rules

### 1. Deadline Semantics

- **Deadline validation**: Watering at time `t = dk` is **valid** (use `≤`, not `<`)
- **Time progression**: Each action consumes exactly one time step
- **Order**: Apply action → update state → increment time → check refill

### 2. Watering Rules

- **Valid watering**: Can only `W` on a cell that currently has an **unwatered plant**
- **Water capacity check**: Before watering, ensure `current_water ≥ wk`
- **Cell-specific tracking**: Mark that **specific cell** as watered (not just the plant type)
- **Water consumption**: Subtract `wk` from current water after successful watering

### 3. Refill Mechanics

- **Refill trigger**: Refill to `W0` when robot **arrives** at (0,0) after a move action
- **No refill on**: `W` actions or while stationary at (0,0)

### 4. Plan Validation

- **Valid alphabet**: Plan must contain only `{U, D, L, R, W}`
- **Normalization**: Recommend `.strip()` whitespace and `.upper()` case
- **Rejection**: Any invalid characters should cause validation failure

### 5. Movement Mapping

- **Direction vectors**:
  - `U = (-1, 0)` (up = decrease row)
  - `D = (+1, 0)` (down = increase row)
  - `L = (0, -1)` (left = decrease col)
  - `R = (0, +1)` (right = increase col)
- **Boundary checking**: Reject moves that go out of bounds
- **Obstacle checking**: Reject moves into cells with value `-1`

### 6. Complete Plant Coverage

- **Individual cells**: Must water **every plant cell**, not just one per type
- **Multiple instances**: Same plant type can appear in multiple cells
- **Tracking**: Use `set[(r,c)]` to track watered positions

### 7. Input Parsing Rules

- **Grid consistency**: Compute `K = max(grid)` (max positive value)
- **Type definitions**: Read exactly `K` lines of `(wk, dk)` pairs
- **Origin constraint**: Enforce that `(0,0)` is always `0` (empty)
- **Comment handling**: Skip lines starting with `#` and blank lines everywhere
- **Row/column validation**: Enforce exactly `N` rows and `M` items per row; reject otherwise
- **Label bounds**: Every grid cell must be `-1`, `0`, or in `1..K` where `K = max(grid)`
- **Zero plants case**: If `K=0` (no plants), accept `0` type definition lines
- **File handle**: Never call `close()` on `fh` in `load()` (harness manages it)

### 8. Plan Processing

- **Normalization**: Apply `plan = plan.strip().upper()` before validation
- **Character validation**: Reject if any character not in `{U,D,L,R,W}` remains
- **Timing precision**: Increment time after action, then check `time ≤ dk` for watering
- **Refill timing**: Only refill after move actions that place robot at `(0,0)`
- **Cell tracking**: Track watered cells by coordinates `set[(r,c)]`, not by plant type

### 9. Return Values

- **check_solution return**: Must return `bool` only
- **Verbosity**: If `verbose=True`, print diagnostics but still return `bool`
- **No origin return**: Don't require robot to end at (0,0)

### 10. Edge Cases & Robustness

- **Empty plants case**: If `K=0` (no plants), empty plan `""` is valid; any non-empty plan must still follow movement rules
- **K consistency**: If grid's max label is `Kgrid`, must read exactly `Kgrid` type lines; reject if more/fewer appear
- **Value domains**: Enforce integers everywhere; reject malformed tokens; accept `W0 ≥ 0` (don't assume `> 0`)
- **Exact water consumption**: On `W` action, subtract exactly `wk`; never allow negative water levels
- **Single-pass validation**: Don't pre-scan or modify plan beyond `.strip().upper()`; validate streaming to avoid O(n²) logic
- **Standard library only**: No external imports beyond `search.py` (provided by grader); use only built-in Python libraries

### 11. Final Micro-Clarifications

- **No optimality check**: `check_solution` validates correctness only; must not require shortest plans; extra moves after all plants watered are allowed if valid
- **Watering doesn't move**: `W` action only affects water/time/plant state; robot position stays the same
- **Finish condition**: Plan valid iff every plant cell watered exactly once within deadline—return `True` even if plan continues with valid moves
- **Parsing tolerance**: Accept arbitrary spacing (multiple spaces/tabs) and trailing spaces; still enforce strict integer tokens
- **Verbose contract**: Only print diagnostics when `verbose=True`; always return a `bool`

## Time Update Order

```
for each action in plan:
    1. Apply action (move/water)
    2. Update state (position/water/plants)
    3. time += 1
    4. If at (0,0) after a move, refill to W0
```

## Data Structures to Track

```python
watered = set()           # set[(r,c)] - watered plant positions
plants = {}               # dict[(r,c)] -> plant_type
types = {}                # dict[plant_type] -> (wk, dk)
```

## Expected Input/Output Examples

Based on the `public1/` test files, our system must handle:

**Example Input (ex0.dat):**

```
3 4 100                   # 3 rows, 4 columns, 100 water capacity
0  0 1  0                # Grid row 1
0 -1 2 -1                # Grid row 2 (obstacles and plants)
1  0 1 -1                # Grid row 3
2 10                     # Plant type 1: needs 2 water, deadline 10
3 15                     # Plant type 2: needs 3 water, deadline 15
```

**Expected Output (ex0.plan):**

```
DDWRRWUWUW              # Action sequence: Down,Down,Water,Right,Right,Water,Up,Water,Up,Water
```

## Validation Requirements

### Positive Tests (Must Return True)

All solutions must pass the 10 test cases in `public1/`:

- `ex0.dat` → `ex0.plan`
- `ex1.dat` → `ex1.plan`
- `ex2.dat` → `ex2.plan`
- `ex3.dat` → `ex3.plan`
- `ex4.dat` → `ex4.plan`
- `ex5.dat` → `ex5.plan`
- `ex6.dat` → `ex6.plan`
- `ex7.dat` → `ex7.plan`
- `ex8.dat` → `ex8.plan`
- `ex9.dat` → `ex9.plan`

### Negative Tests (Must Return False)

Your `check_solution()` must correctly reject invalid plans:

**Invalid Characters:**

- Plan containing illegal characters (e.g., `"DDWRXUW"` with `X`)
- Plans with lowercase letters if not normalized
- Plans with whitespace if not stripped

**Invalid Watering:**

- `W` action on empty cell (value `0`)
- `W` action on obstacle cell (value `-1`)
- `W` action on already watered plant cell
- `W` action with insufficient water (`current_water < wk`)

**Invalid Movement:**

- Move that goes out of grid boundaries
- Move into obstacle cell (value `-1`)

**Constraint Violations:**

- Missing any plant cell (not watering all required plants)
- Watering past deadline (`time > dk` when watering)
- Plan that leaves plants unwatered
- Watering the same cell twice
- Moving out of grid boundaries
- Moving into obstacle cells (value `-1`)

**Edge Cases:**

- Empty plan string `""`
- Plan with only invalid actions

### Additional Sanity Checks

**Critical Rejections:**

- Reject plans that leave any plant unwatered
- Reject plans that water any cell twice  
- Reject plans that water with insufficient water
- Reject plans that move out of bounds
- Reject plans that move into obstacles (`-1` cells)
- Reject plans that miss any deadline

**Critical Acceptances:**

- Accept plans that water exactly at deadline (`time ≤ dk`, not `time < dk`)
- Accept plans with valid normalization after `.strip().upper()`

---

# Success Criteria

## Essential Requirements

- ✅ Complete `GardenerProblem` class implementation
- ✅ Working `load()` method handling all input formats
- ✅ Working `check_solution()` method with comprehensive validation
- ✅ All 10 official test cases pass
- ✅ Basic edge case handling (empty plans, invalid actions, etc.)

## Submission Requirements

- ✅ Single file `solution.py` submitted to Moodle
- ✅ Code passes all official tests
- ✅ Clean, readable code with basic documentation
- ✅ Submitted before September 26, 2025 deadline

This lean approach focuses on the essential deliverables while maintaining quality through systematic testing with the provided official test cases. The timeline provides adequate buffer for testing and integration without unnecessary complexity.
