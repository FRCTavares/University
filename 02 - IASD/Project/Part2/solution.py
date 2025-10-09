###############################################################################################
# Autonomous gardening robot for a sustainable Mars settlement                  #
#                                                                                             #
# Group 7                                                                                     #
#                                                                                             #
# Members:                                                                                    #
#   Diogo Sampaio (103068)                                                                    #
#   Francisco Tavares (103402)                                                                #
#   Marta Valente (103574)                                                                    #
###############################################################################################

import time
import search
from pathlib import Path

# ============================================================================
# GardenerProblem: Autonomous Mars Gardening Robot
# ============================================================================
# Implements an uninformed search problem for a robot that must:
# - Navigate a grid from origin (0,0)
# - Water plants before their deadlines
# - Refill water at origin when needed
# - Avoid obstacles
# ============================================================================

class GardenerProblem(search.Problem):
    """
    Mars gardening robot problem using uninformed breadth-first search.
    
    State representation: (robot_pos, water_level, time, watered_plants)
    - robot_pos: (x,y) coordinates where x=column, y=row
    - water_level: current water amount (integer)
    - time: elapsed time units (integer)
    - watered_plants: bitmask where bit i indicates if plant i is watered
    """
    
    def __init__(self):
        self.map = []              # 2D grid: map[row][col], values: 0=empty, -1=obstacle, 1..K=plant type
        self.water_capacity = None # Maximum water capacity W0 of the robot
        self.plants = {}           # (x,y) -> (plant_type, water_needed, deadline)
        self.obstacles = set()     # Set of (x,y) obstacle positions
        self.plant_types = {}      # plant_type_id -> (water_needed, deadline)
        self.visited_states = {}   # State dominance tracking: (pos, watered_plants) -> [(time, water), ...]
        
    def load(self, fh):
        """
        Load problem instance from input file.
        
        File format:
        - Line 1: N M W0 (grid height, width, initial water capacity)
        - Lines 2 to N+1: Grid rows (0=empty, -1=obstacle, 1..K=plant type)
        - Lines N+2 to N+K+1: Plant type definitions 'wk dk' (water needed, deadline)
        
        Args:
            fh: File handle to read from
            
        Raises:
            ValueError: If file format is invalid or constraints are violated
        """
        # Read all non-empty, non-comment lines from input file
        lines = []
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        if not lines:
            raise ValueError("No valid lines found in file")

        # Parse header: N (height), M (width), W0 (initial water capacity)
        tokens = lines[0].split()
        if len(tokens) != 3:
            raise ValueError(f"Expected 3 tokens on first line, got {len(tokens)}")
        N, M, W0 = int(tokens[0]), int(tokens[1]), int(tokens[2])
        if N <= 0 or M <= 0:
            raise ValueError(f"Invalid grid dimensions: N={N}, M={M}")
        self.water_capacity = W0

        # Parse grid (next N lines)
        if len(lines) < 1 + N:
            raise ValueError(f"Expected {N} grid rows, only {len(lines) - 1} available")
        self.map = []
        for i in range(1, N + 1):
            row = [int(x) for x in lines[i].split()]
            if len(row) != M:
                raise ValueError(f"Malformed grid: row {i} has {len(row)} columns, expected {M}")
            self.map.append(row)
        if len(self.map) != N:
            raise ValueError(f"Malformed grid: expected {N} rows, got {len(self.map)}")

        # Validate: origin (0,0) must be empty for robot to start
        if self.map[0][0] != 0:
            raise ValueError("Invalid input: origin (0,0) must be empty (0).")

        # Determine K (number of plant types) = max plant type ID in grid
        K = 0
        for r in range(N):
            for c in range(M):
                v = self.map[r][c]
                if v > 0:  # Plant type found
                    if v > K:
                        K = v

        # Parse plant type definitions (K lines after the grid)
        plant_def_start = 1 + N
        if len(lines) < plant_def_start + K:
            raise ValueError(f"Expected {K} plant type definitions, only {len(lines) - plant_def_start} available")

        self.plant_types = {}
        for i in range(K):
            parts = lines[plant_def_start + i].split()
            if len(parts) != 2:
                raise ValueError(f"Plant type {i+1} definition malformed: {lines[plant_def_start + i]}")
            wk, dk = int(parts[0]), int(parts[1])  # wk=water needed, dk=deadline
            self.plant_types[i + 1] = (wk, dk)

        # Scan grid to populate obstacles and plant instances
        self.obstacles = set()
        self.plants = {}   # (x,y) -> (plant_type, water_needed, deadline)
        for r in range(N):
            for c in range(M):
                cell = self.map[r][c]
                pos = (c, r)  # Position as (x=column, y=row)
                if cell == -1:
                    self.obstacles.add(pos)
                elif cell > 0:  # Plant cell
                    if cell not in self.plant_types:
                        raise ValueError(f"Plant type {cell} at {pos} not defined in type table")
                    wk, dk = self.plant_types[cell]
                    self.plants[pos] = (cell, wk, dk)

        # Store grid dimensions for boundary checking
        self.H = N  # Height (number of rows)
        self.W = M  # Width (number of columns)

        # Index plants with unique IDs for bitmask representation
        # Plants are indexed in row-major order for deterministic ordering
        self.plant_idx = {}   # (x,y) -> unique index [0..P-1]
        self.wk = []          # Water needed for plant at index i
        self.dk = []          # Deadline for plant at index i

        idx = 0
        for r in range(N):
            for c in range(M):
                pos = (c, r)
                if pos in self.plants:
                    ptype, pwk, pdk = self.plants[pos]
                    self.plant_idx[pos] = idx
                    self.wk.append(pwk)
                    self.dk.append(pdk)
                    idx += 1
        
        self.P = idx  # Total number of plants
        if self.P == 0:
            raise ValueError("No plants found in grid.")

        # Initial state: robot at origin (0,0) with full water, time=0, no plants watered
        self.initial = ((0, 0), self.water_capacity, 0, 0)
        super().__init__(self.initial, None)

    def prune(self, pos, mask, time, water):
        """
        State dominance pruning to eliminate worse states.
        
        For states at the same (position, watered_plants), we only keep states that
        are not dominated. A state is dominated if another state exists with:
        - Earlier or equal time AND
        - Greater or equal water
        
        This eliminates states that can never lead to better solutions, significantly
        reducing the search space.
        
        Args:
            pos: Robot position (x, y)
            mask: Bitmask of watered plants
            time: Current time
            water: Current water level
            
        Returns:
            True if state should be pruned (dominated), False otherwise
        """
        # Get or create list of non-dominated states for this (position, watered_plants)
        frontier = self.visited_states.setdefault((pos, mask), [])
        
        # Check if current state is dominated by any state in frontier
        for (t, w) in frontier:
            if t <= time and w >= water:
                return True  # State is dominated -> prune it

        # Remove states from frontier that are dominated by the current state
        keep = []
        for (t, w) in frontier:
            if not (time <= t and water >= w):  # Not dominated by current state
                keep.append((t, w))
        
        # Add current state to frontier
        keep.append((time, water))
        frontier[:] = keep
        return False  # State is not dominated -> don't prune

    def actions(self, state):
        """
        Generate legal actions from the current state with multiple optimizations.
        
        Optimizations applied:
        1. Deadline pruning: Prune if any unwatered plant deadline has passed
        2. State dominance pruning: Eliminate states dominated by better states
        3. Neighbor plant prioritization: Prefer moves toward waterable plants
        
        Args:
            state: Tuple (robot_pos, water_level, time, watered_plants)
            
        Returns:
            List of legal actions ['W', 'L', 'R', 'U', 'D'], or empty list if branch should be pruned
        """
        robot_pos, water_level, time, watered_plants = state

        # ================================================================
        # OPTIMIZATION 1: Deadline Pruning
        # ================================================================
        # If any unwatered plant's deadline has already passed, prune this branch
        for _, idx in self.plant_idx.items():  
            if ((watered_plants >> idx) & 1) == 0 and time > self.dk[idx]:
                return []  # Deadline missed -> impossible to complete

        # ================================================================
        # OPTIMIZATION 2: State Dominance Pruning
        # ================================================================
        # Check if this state is dominated by another with same (pos, watered_plants)
        if self.prune(robot_pos, watered_plants, time, water_level):
            return []  # Dominated state -> prune

        possible_actions = []
        x, y = robot_pos

        # ================================================================
        # WATERING ACTION
        # ================================================================
        # Check if we're on a plant that can be watered
        i = self.plant_idx.get(robot_pos, None)
        if i is not None:
            not_watered = (((watered_plants >> i) & 1) == 0)
            # Can water if: plant exists, not watered, enough water, within deadline
            if not_watered and water_level >= self.wk[i] and time <= self.dk[i]:
                possible_actions.append('W')

        # ================================================================
        # MOVEMENT ACTIONS WITH NEIGHBOR PLANT PRIORITIZATION
        # ================================================================
        # Generate all possible moves: Left, Right, Up, Down
        h, w = self.H, self.W
        moves = [
            ('L', x - 1, y),  # Left
            ('R', x + 1, y),  # Right
            ('U', x, y - 1),  # Up
            ('D', x, y + 1)   # Down
        ]

        # Separate moves into priority tiers for better search efficiency
        priority_moves = []  # Moves toward waterable adjacent plants
        regular_moves = []   # All other valid moves
        
        for action, nx, ny in moves:
            # Only consider moves within grid bounds and not into obstacles
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in self.obstacles:
                # Check if destination has an unwatered plant we could water
                neighbor_idx = self.plant_idx.get((nx, ny), None)
                if neighbor_idx is not None:
                    is_watered = ((watered_plants >> neighbor_idx) & 1) == 1
                    if not is_watered:
                        # Check if we'll have enough water and time after moving
                        if water_level >= self.wk[neighbor_idx] and time + 1 <= self.dk[neighbor_idx]:
                            # This move leads to a plant we can water immediately -> prioritize it
                            priority_moves.append(action)
                            continue
                
                # Regular move (valid but not prioritized)
                regular_moves.append(action)
        
        # Add priority moves first to explore promising paths earlier in BFS
        possible_actions.extend(priority_moves)
        possible_actions.extend(regular_moves)

        return possible_actions
    
    def result(self, state, action):
        """
        Apply an action to a state and return the resulting state.
        
        Actions:
        - 'W': Water the plant at current position (reduces water, marks plant as watered)
        - 'L'/'R'/'U'/'D': Move left/right/up/down (refills water if arriving at origin)
        
        Important: Watering completes at time t+1 (not t), and water refill only
        occurs when ARRIVING at origin via movement (not during watering or idling).
        
        Args:
            state: Current state (robot_pos, water_level, time, watered_plants)
            action: Action to apply ('W', 'L', 'R', 'U', or 'D')
            
        Returns:
            New state after applying the action
            
        Raises:
            ValueError: If action is invalid for the current state
        """
        (x, y), water_level, time, watered_plants = state
        
        # All actions take 1 time unit
        new_time = time + 1

        # ================================================================
        # WATERING ACTION
        # ================================================================
        if action == 'W':
            i = self.plant_idx.get((x, y), None)
            # Validate: must be on a plant that hasn't been watered yet
            if i is None or ((watered_plants >> i) & 1) == 1:
                raise ValueError(f"Invalid water action at {(x,y)}")
            
            # Consume water and mark plant as watered (set bit i in bitmask)
            new_water_level = water_level - self.wk[i]
            new_watered_plants = watered_plants | (1 << i)
            return ((x, y), new_water_level, new_time, new_watered_plants)

        # ================================================================
        # MOVEMENT ACTIONS
        # ================================================================
        # Calculate new position based on movement direction
        if action == 'L': 
            new_pos = (x - 1, y)
        elif action == 'R': 
            new_pos = (x + 1, y)
        elif action == 'U': 
            new_pos = (x, y - 1)
        elif action == 'D': 
            new_pos = (x, y + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Validate movement is within bounds and not into obstacle
        nx, ny = new_pos
        if not (0 <= nx < self.W and 0 <= ny < self.H) or new_pos in self.obstacles:
            raise ValueError(f"Invalid move {action} from {(x,y)} to {new_pos}")
        
        # Water refill: only when ARRIVING at origin (0,0) via movement
        new_water_level = self.water_capacity if new_pos == (0, 0) else water_level

        return (new_pos, new_water_level, new_time, watered_plants)

    def path_cost(self, c, state1, action, state2):
        """
        Calculate the cost of taking an action from state1 to state2.
        
        Each action (watering or movement) costs 1 time unit.
        
        Args:
            c: Cost of path to state1
            state1: Starting state
            action: Action taken
            state2: Resulting state
            
        Returns:
            Total path cost to state2
        """
        return c + 1  # Each action costs 1 time unit

    def goal_test(self, state):
        """
        Test if all plants have been watered (goal condition).
        
        A state is a goal when all P plants are watered, which means
        all bits in the watered_plants bitmask are set to 1.
        
        Args:
            state: State to test (robot_pos, water_level, time, watered_plants)
            
        Returns:
            True if all plants are watered, False otherwise
        """
        _, _, _, watered_plants = state
        # All plants watered when bitmask has all P bits set: 2^P - 1
        return watered_plants == (1 << self.P) - 1 

    def solve(self):
        """
        Solve the gardening problem using breadth-first tree search.
        
        Returns:
            String of actions (e.g., "RRWDLW") if solution found, None otherwise
        """
        print("Initializing search...")
        self.visited_states = {}  # Reset state tracking
        
        start_time = time.time()
        
        # Use breadth_first_tree_search from search.py
        result = search.breadth_first_tree_search(self)
        
        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.3f} seconds.")
        
        if result:
            return "".join(result.solution())  # Convert action list to string
        return None
            

# ============================================================================
# MAIN: Test Suite
# ============================================================================
# Tests all examples (ex0-ex9) and displays performance metrics
# ============================================================================

if __name__ == "__main__":
    # Test all examples ex0-ex9
    print("Testing all examples ex0-ex9 with individual timing")
    print("=" * 60)
    
    total_start_time = time.time()
    results = []
    
    for i in range(10):
        example_to_test = f"ex{i}"
        dat_file = Path(f"public2/{example_to_test}.dat")

        if not dat_file.exists():
            print(f"File {dat_file} not found! Skipping...")
            continue
        
        print(f"\nSolving for {example_to_test}")
        print("-" * 40)
        
        try:
            # Create problem instance
            problem = GardenerProblem()
            
            # Load the problem from file
            with open(dat_file, 'r') as f:
                problem.load(f)
            print("Problem loaded successfully.")

            # Solve the problem with individual timing
            individual_start = time.time()
            plan = problem.solve()
            individual_end = time.time()
            individual_time = individual_end - individual_start

            # Display results
            if plan is not None:
                print(f"Found a valid path plan with {len(plan)} actions: {plan}")
                print(f"Individual solve time: {individual_time:.3f} seconds")
                results.append((example_to_test, len(plan), individual_time, "SUCCESS"))
            else:
                print("No valid plan found.")
                print(f"Individual solve time: {individual_time:.3f} seconds")
                results.append((example_to_test, 0, individual_time, "FAILED"))

        except Exception as e:
            print(f"Error testing {example_to_test}: {e}")
            results.append((example_to_test, 0, 0, f"ERROR: {e}"))
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # ========================================================================
    # Summary Table
    # ========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL TESTS")
    print("=" * 60)
    print(f"{'Example':<8} {'Actions':<8} {'Time(s)':<10} {'Status':<15}")
    print("-" * 60)
    
    for example, actions, solve_time, status in results:
        print(f"{example:<8} {actions:<8} {solve_time:<10.3f} {status:<15}")
    
    print("-" * 60)
    print(f"Total execution time: {total_time:.3f} seconds")
    print(f"Tests completed: {len(results)}")
    successful = sum(1 for _, _, _, status in results if status == "SUCCESS")
    print(f"Successful: {successful}/{len(results)}")
    print("=" * 60)
