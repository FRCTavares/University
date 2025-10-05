###############################################################################################
# Assignment 1: Autonomous gardening robot for a sustainable Mars settlement                  #
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
        Load problem from file with the correct format:
        Line 1: N M W0 (rows, cols, water capacity)
        Next N lines: The map (0=empty, -1=obstacle, 1,2,3...=plant types)
        K is computed as max(cell) over grid where cell > 0
        Next K lines: wk dk (water needed, deadline for each plant type)
        
        Args:
            fh: File handle (already opened file pointer)
        """
        # Read all lines, stripping whitespace and ignoring comments/blank lines
        lines = []
        for line in fh:
            line = line.strip()
            # Skip blank lines and comments
            if line and not line.startswith('#'):
                lines.append(line)
        
        if not lines:
            raise ValueError("No valid lines found in file")
        
        # Parse first line: N M W0
        first_line = lines[0].split()
        if len(first_line) != 3:
            raise ValueError(f"Expected 3 tokens on first line, got {len(first_line)}")
        N, M, W0 = int(first_line[0]), int(first_line[1]), int(first_line[2])
        self.water_capacity = W0
        
        if N <= 0 or M <= 0:
            raise ValueError(f"Invalid grid dimensions: N={N}, M={M}")
        
        # Parse the map (next N lines)
        if len(lines) < 1 + N:
            raise ValueError(f"Expected {N} grid rows, only {len(lines) - 1} available")
        
        self.map = []
        for i in range(1, N + 1):
            row = [int(x) for x in lines[i].split()]
            if len(row) != M:
                raise ValueError(f"Malformed grid: row {i} has {len(row)} columns, expected {M}")
            self.map.append(row)
        
        # Verify exactly N rows
        if len(self.map) != N:
            raise ValueError(f"Malformed grid: expected {N} rows, got {len(self.map)}")
        
        # Enforce origin (0,0) must be empty
        if self.map[0][0] != 0:
            raise ValueError("Invalid input: origin (0,0) must be empty (0).")        # Compute K = max plant type in grid
        K = 0
        for row in self.map:
            for cell in row:
                if cell > 0:
                    K = max(K, cell)

        # Parse plant type definitions (K lines after the map)
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

        # Find all plants in the map and their positions
        # Store as (x,y) → (plant_type, wk, dk) for fast lookup
        self.plants = {}
        self.obstacles = set()
        
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                cell_value = self.map[row][col]
                pos = (col, row)  # (x, y) format
                if cell_value == -1:
                    # Obstacle
                    self.obstacles.add((col, row))  # (x, y) format
                elif cell_value > 0:
                    # Plant of type cell_value
                    if cell_value not in self.plant_types:
                        raise ValueError(f"Plant type {cell_value} at {pos} not defined")
                    wk, dk = self.plant_types[cell_value]
                    self.plants[pos] = (cell_value, wk, dk)

        # Initial state: robot at (0,0), full water, time=0, no plants watered
        initial_state = ((0, 0), self.water_capacity, 0, frozenset())
        self.initial = initial_state
        super().__init__(self.initial, None)
    
    def _dominance_prune(self, robot_pos, watered_plants, time, water_limit, cap=6):
        """
        Skyline dominance on (time, water) for a given (robot_pos, watered_plants).
        Keep a small set of non-dominated pairs to stay efficient.
        Return True if (time, water) is dominated and can be pruned.
        """
        key = (robot_pos, watered_plants)
        visited_states = self.visited_states.get(key)
        if visited_states is None:
            # store as list of tuples
            self.visited_states[key] = [(time, water_limit)]
            return False

        # dominated by any?
        for (by_time, by_water) in visited_states:
            if (by_time <= time and by_water >= water_limit) and (by_time < time or by_water > water_limit):
                return True  # dominated -> prune

        # not dominated -> insert and clean up dominated entries
        visited_states.append((time, water_limit))

        # remove entries dominated by the new one
        new_visited_states = []
        for (by_time, by_water) in visited_states:
            # keep (by_time, by_water) only if NOT dominated by (time, water)
            if not ((time <= by_time and water_limit >= by_water) and (time < by_time or water_limit > by_water)):
                new_visited_states.append((by_time, by_water))

        # Keep the “most promising” few entries only
        new_visited_states.sort(key=lambda p: (p[0], -p[1]))
        if len(new_visited_states) > cap:
            new_visited_states = new_visited_states[:cap]
        self.visited_states[key] = new_visited_states
        return False

    def actions(self, state):
        """
        State: (robot_pos, water_level, time, watered_plants)
        Return: list of actions in {'W','L','R','U','D'} that are legal.
        """
        robot_pos, water_level, time, watered_plants = state

        # --- Watering ---
        # If we can water now, we should expose 'W' even if dominated,
        # because earlier recorded states might not have had enough water.
        possible_actions = []
        if robot_pos in self.plants and robot_pos not in watered_plants:
            _, wk, dk = self.plants[robot_pos]
            if water_level >= wk and time <= dk:
                possible_actions.append('W')

        # Only after checking immediate 'W' do skyline pruning
        if self._dominance_prune(robot_pos, watered_plants, time, water_level):
            return possible_actions  # may already contain 'W'; otherwise empty

        x, y = robot_pos
        h = len(self.map)
        w = len(self.map[0]) if h > 0 else 0

        # --- Possible Moves ---
        moves = (('L', (x - 1, y)),
                ('R', (x + 1, y)),
                ('U', (x, y - 1)),
                ('D', (x, y + 1)))

        # Filter out invalid moves
        for a, (nx, ny) in moves:
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in self.obstacles:
                possible_actions.append(a)

        return possible_actions
    
    def result(self, state, action):
        """Return the resulting state after applying the action to the given state."""
        robot_pos, water_level, time, watered_plants = state
        height = len(self.map)
        width = len(self.map[0]) if height > 0 else 0

        # Initialize new state variables
        new_robot_pos = robot_pos
        new_water_level = water_level
        new_watered_plants = watered_plants

        # Movement actions
        if action == 'L':
            new_robot_pos = (robot_pos[0] - 1, robot_pos[1])
        elif action == 'R':
            new_robot_pos = (robot_pos[0] + 1, robot_pos[1])
        elif action == 'U':
            new_robot_pos = (robot_pos[0], robot_pos[1] - 1)
        elif action == 'D':
            new_robot_pos = (robot_pos[0], robot_pos[1] + 1)
        elif action == 'W':
            if robot_pos not in self.plants or robot_pos in watered_plants:
                raise ValueError(f"Invalid water action at {robot_pos}")

            _, wk, dk = self.plants[robot_pos]
            new_water_level = water_level - wk
            new_watered_plants = watered_plants.union((robot_pos,))

        # Check if back at origin - refill water (for movement actions)
        if action in ['L', 'R', 'U', 'D']:
            nx, ny = new_robot_pos
            if not (0 <= nx < width and 0 <= ny < height) or new_robot_pos in self.obstacles:
                raise ValueError(f"Invalid move {action} from {robot_pos} to {new_robot_pos}")

            # Refill water if moved into origin
            if new_robot_pos == (0, 0):
                new_water_level = self.water_capacity

        # Increment time for any action
        new_time = time + 1

        return (new_robot_pos, new_water_level, new_time, new_watered_plants)
    
    def path_cost(self, c, state1, action, state2):
        """Calculate the cost of a path from state1 to state2."""
        return c + 1  # Each action costs 1 time unit

    def goal_test(self, state):
        """Goal: all plants are watered."""
        _, _, _, watered_plants = state
        return len(watered_plants) == len(self.plants)

    def solve(self):
        """Solve the problem with an uninformed search algorithm."""
        print("Initializing search...")
        self.visited_states = {}
        solution_node = search.uniform_cost_search(self)
        return "".join(solution_node.solution()) if solution_node else None
            
if __name__ == "__main__":
    # Which example to test
    example_to_test = "ex9"  
    
    # Test the specified example
    dat_file = Path(f"public2/{example_to_test}.dat")

    if not dat_file.exists():
        print(f"File {dat_file} not found!")
        exit(1)
    
    print(f"\nSolving for {example_to_test}")
    print("=" * 50)
    
    try:
        # Create problem instance
        problem = GardenerProblem()
        
        # Load the problem with file handle
        with open(dat_file, 'r') as f:
            problem.load(f)
        print("Problem loaded successfully.")

        # Solve the problem
        plan = problem.solve()

        # Print results
        if plan is not None:
            print(f"Found a valid path plan with {len(plan)} actions: {plan}\n")
        else:
            print("No valid plan found.\n")

    except Exception as e:
        print(f"Error testing {example_to_test}: {e}")