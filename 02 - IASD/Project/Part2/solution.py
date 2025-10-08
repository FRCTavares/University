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

# Define the problem class
class GardenerProblem(search.Problem):
    def __init__(self):
        self.map = []              # 2D list representing the map
        self.water_capacity = None # Maximum water capacity of the robot
        self.plants = {}           # Maps (x,y) positions to (plant_type, wk, dk)
        self.obstacles = set()     # Set of (x,y) positions of obstacles
        self.plant_types = {}      # Maps plant type number to (water_needed, deadline)
        self.visited_states = {}   # Dictionary to track (pos, water, watered_plants) -> best_time
        
    def load(self, fh):
        """
        Load problem from file with the format:
        Line 1: N M W0
        Next N lines: grid (0 empty, -1 obstacle, 1..K plant type)
        Next K lines: 'wk dk' per plant type id (1..K)
        """
        # --- Read all non-empty, non-comment lines ---
        lines = []
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        if not lines:
            raise ValueError("No valid lines found in file")

        # --- Header: N, M, W0 ---
        tokens = lines[0].split()
        if len(tokens) != 3:
            raise ValueError(f"Expected 3 tokens on first line, got {len(tokens)}")
        N, M, W0 = int(tokens[0]), int(tokens[1]), int(tokens[2])
        if N <= 0 or M <= 0:
            raise ValueError(f"Invalid grid dimensions: N={N}, M={M}")
        self.water_capacity = W0

        # --- Grid (next N lines) ---
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

        # Origin must be empty
        if self.map[0][0] != 0:
            raise ValueError("Invalid input: origin (0,0) must be empty (0).")

        # --- Determine K (max plant type id present in grid) ---
        K = 0
        for r in range(N):
            for c in range(M):
                v = self.map[r][c]
                if v > 0:
                    if v > K:
                        K = v

        # --- Plant type table (K lines after the grid) ---
        plant_def_start = 1 + N
        if len(lines) < plant_def_start + K:
            raise ValueError(f"Expected {K} plant type definitions, only {len(lines) - plant_def_start} available")

        self.plant_types = {}
        for i in range(K):
            parts = lines[plant_def_start + i].split()
            if len(parts) != 2:
                raise ValueError(f"Plant type {i+1} definition malformed: {lines[plant_def_start + i]}")
            wk, dk = int(parts[0]), int(parts[1])
            self.plant_types[i + 1] = (wk, dk)

        # --- Scan grid to build obstacles and per-plant instances ---
        self.obstacles = set()
        self.plants = {}   # (x,y) -> (plant_type, wk, dk)
        for r in range(N):
            for c in range(M):
                cell = self.map[r][c]
                pos = (c, r)  # (x,y)
                if cell == -1:
                    self.obstacles.add(pos)
                elif cell > 0:
                    if cell not in self.plant_types:
                        raise ValueError(f"Plant type {cell} at {pos} not defined in type table")
                    wk, dk = self.plant_types[cell]
                    self.plants[pos] = (cell, wk, dk)

        # --- Grid helpers ---
        self.H = N
        self.W = M

        # --- Index plants for bitmask/state speed (NOW self.plants is populated) ---
        self.plant_idx = {}   # (x,y) -> idx [0..P-1]
        self.wk = []
        self.dk = []

        # Deterministic order: row-major over the grid so indexes are stable
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
        self.P = idx
        if self.P == 0:
            raise ValueError("No plants found in grid.")

        # --- Initial state ---
        self.initial = ((0, 0), None , self.water_capacity, 0, 0)
        super().__init__(self.initial, None)

    def prune(self, pos, mask, time, water):
        """
        Keep a Pareto frontier over (time, water) per (pos, mask).
        Prune if there exists (t*, w*) with t* <= time and w* >= water.
        """
        frontier = self.visited_states.setdefault((pos, mask), [])
        for (t, w) in frontier:
            if t <= time and w >= water:
                return True  # dominated -> prune

        # keep only records not worse than the new one
        keep = []
        for (t, w) in frontier:
            if not (time <= t and water >= w):   
                keep.append((t, w))
        keep.append((time, water))
        frontier[:] = keep
        return False

    def actions(self, state):
        """
        State: (robot_pos, water_level, time, watered_plants)
        Return: list of actions in {'W','L','R','U','D'} that are legal.
        """
        robot_pos, prev_pos, water_level, time, watered_plants = state

        # --- Deadline Pruning ---
        for _, idx in self.plant_idx.items():  
            if ((watered_plants >> idx) & 1) == 0 and time > self.dk[idx]:
                return []

        # --- Dominance Pruning ---
        if self.prune(robot_pos, watered_plants, time, water_level):
            return []

        possible_actions = []

        # --- Watering ---
        i=self.plant_idx.get(robot_pos, None)

        # Possible water action
        if i is not None:
            not_watered = (((watered_plants >> i) & 1) == 0)
            if not_watered and water_level >= self.wk[i] and time <= self.dk[i]:
                possible_actions.append('W')

        x, y = robot_pos
        h, w = self.H, self.W

        # --- Possible Moves ---
        moves = (('L', (x - 1, y)),
                ('R', (x + 1, y)),
                ('U', (x, y - 1)),
                ('D', (x, y + 1)))

        # Filter out invalid moves
        for a, (nx, ny) in moves:
            np = (nx, ny)
            if 0 <= nx < w and 0 <= ny < h and np not in self.obstacles:
                if prev_pos is None or np != prev_pos:
                    possible_actions.append(a)

        return possible_actions
    
    def result(self, state, action):
        """Return the resulting state after applying the action to the given state."""
        (x, y), prev_pos, water_level, time, watered_plants = state
        
        # Default new state values
        new_pos = (x, y)
        new_prev = prev_pos
        new_water_level = water_level
        new_watered_plants = watered_plants

        if action == 'L': new_pos = (x-1, y)
        elif action == 'R': new_pos = (x+1, y)
        elif action == 'U': new_pos = (x, y-1)
        elif action == 'D': new_pos = (x, y+1)
        elif action == 'W':
            i = self.plant_idx.get((x,y), None)
            if i is None or ((watered_plants >> i) & 1) == 1:
                raise ValueError(f"Invalid water action at {(x,y)}")
            
            new_water_level = water_level - self.wk[i]
            new_watered_plants = watered_plants | (1 << i)
            new_prev = None

        h, w = self.H, self.W

        # Movement validity + refill
        if action in ('L', 'R', 'U', 'D'):
            nx, ny = new_pos
            if not (0 <= nx < w and 0 <= ny < h) or new_pos in self.obstacles:
                raise ValueError(f"Invalid move {action} from {(x,y)} to {new_pos}")
            if new_pos == (0, 0):
                new_water_level = self.water_capacity
            new_prev = (x,y)

        new_time = time + 1

        return (new_pos, new_prev, new_water_level, new_time, new_watered_plants)

    def path_cost(self, c, state1, action, state2):
        """Calculate the cost of a path from state1 to state2."""
        return c + 1  # Each action costs 1 time unit

    def goal_test(self, state):
        """Goal: all plants are watered."""
        _, _, _, _, watered_plants = state
        return watered_plants == (1 << self.P) - 1 

    def solve(self):
        """Solve the problem with an uninformed search algorithm."""
        print("Initializing search...")
        self.visited_states = {}

        start_time = time.time()               
        solution_node = search.breadth_first_graph_search(self)
        end_time = time.time()               
        
        elapsed = end_time - start_time
        print(f"Search completed in {elapsed:.3f} seconds.")

        return "".join(solution_node.solution()) if solution_node else None
            
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
            
            # Load the problem with file handle
            with open(dat_file, 'r') as f:
                problem.load(f)
            print("Problem loaded successfully.")

            # Solve the problem with individual timing
            individual_start = time.time()
            plan = problem.solve()
            individual_end = time.time()
            individual_time = individual_end - individual_start

            # Print results
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
    
    # Summary
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