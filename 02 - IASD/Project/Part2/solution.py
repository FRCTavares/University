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

import search
from pathlib import Path

# Define the problem class
class GardenerProblem(search.Problem):
    def __init__(self):
        self.map = []              # 2D list representing the grid
        self.water_capacity = None # Maximum water capacity of the robot
        self.plants = {}           # Maps (x,y) positions to plant deadlines
        self.obstacles = set()     # Set of (x,y) positions of obstacles
        self.plant_types = {}      # Maps plant type number to (water_needed, deadline)
        self.visited_states = {}   # Dictionary to track (pos, water, watered_plants) -> best_time
        
    def load(self, fh):
        """
        Load problem from file with the correct format:
        Line 1: N M W0 (rows, cols, water capacity)
        Next N lines: The map (0=empty, -1=obstacle, 1,2,3...=plant types)
        Next line: K (number of plant types)
        Next K lines: wk dk (water needed, deadline for each plant type)
        
        Args:
            fh: File handle (already opened file pointer)
        """

        #print(f"Loading problem from file handle")
        
        lines = []
        for line in fh:
            line = line.strip()
            # Skip blank lines and comments
            if line and not line.startswith('#'):
                lines.append(line)
        
        if not lines:
            print("No valid lines found in file")
            return
        
        # Parse first line: N M W0
        first_line = lines[0].split()
        N, M, W0 = int(first_line[0]), int(first_line[1]), int(first_line[2])
        self.water_capacity = W0
        
        # Parse the map (next N lines)
        self.map = []
        for i in range(1, N + 1):
            if i < len(lines):
                row = [int(x) for x in lines[i].split()]
                self.map.append(row)
        
        # Parse plant type definitions (remaining lines after the map)
        self.plant_types = {}
        plant_type_num = 1  # Plant types start from 1
        
        for i in range(N + 1, len(lines)):
            parts = lines[i].split()
            if len(parts) == 2:
                wk, dk = int(parts[0]), int(parts[1])
                self.plant_types[plant_type_num] = (wk, dk)
                plant_type_num += 1

        # Find all plants in the map and their positions
        self.plants = {}
        self.obstacles = set()
        
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                cell_value = self.map[row][col]
                if cell_value == -1:
                    # Obstacle
                    self.obstacles.add((col, row))  # (x, y) format
                elif cell_value > 0:
                    # Plant of type cell_value
                    if cell_value in self.plant_types:
                        water_needed, deadline = self.plant_types[cell_value]
                        self.plants[(col, row)] = deadline  # Store deadline for this plant position

        # Initial state: robot at (0,0), full water, time=0, no watered plants
        initial_state = ((0, 0), self.water_capacity, 0, frozenset())
        super().__init__(initial_state, None)

    def actions(self, state):
        """Return the list of actions that can be executed in the given state.
        State: (robot_pos, water_level, time, watered_plants)
        Returns: List of valid actions ['U', 'D', 'L', 'R', 'W']
        """
        robot_pos, water_level, time, watered_plants = state
        possible_actions = []
        
        # Get map dimensions
        if not self.map:
            return possible_actions
        
        height = len(self.map)
        width = len(self.map[0]) if height > 0 else 0
        x, y = robot_pos
        
        # Check movement actions (U, D, L, R)
        # Only if they don't go outside the grid or into obstacles
        
        # Move Left
        if x > 0 and (x - 1, y) not in self.obstacles:
            possible_actions.append('L')
        
        # Move Right
        if x < width - 1 and (x + 1, y) not in self.obstacles:
            possible_actions.append('R')
        
        # Move Up
        if y > 0 and (x, y - 1) not in self.obstacles:
            possible_actions.append('U')
        
        # Move Down
        if y < height - 1 and (x, y + 1) not in self.obstacles:
            possible_actions.append('D')
        
        # Check water action (W)
        # Only if there's an unwatered plant at current position, sufficient water, and deadline hasn't passed
        if robot_pos in self.plants and robot_pos not in watered_plants:
            # Get plant info
            plant_deadline = self.plants[robot_pos]
            
            # Find plant type to get water requirement
            map_row, map_col = y, x  # Convert (x,y) to (row,col)
            if 0 <= map_row < len(self.map) and 0 <= map_col < len(self.map[map_row]):
                plant_type = self.map[map_row][map_col]
                
                if plant_type in self.plant_types:
                    water_needed, _ = self.plant_types[plant_type]
                    
                    # Check if enough water and deadline hasn't passed
                    if water_level >= water_needed and time <= plant_deadline:
                        possible_actions.append('W')
        
        return possible_actions

    def result(self, state, action):
        """Return the resulting state after applying the action to the given state."""
        robot_pos, water_level, time, watered_plants = state
        
        # Initialize new state variables
        new_robot_pos = robot_pos
        new_water_level = water_level
        new_watered_plants = watered_plants
        new_time = time + 1  # Move this up - time always increments

        if action == 'U':
            new_robot_pos = (robot_pos[0], robot_pos[1] - 1)
        elif action == 'D':
            new_robot_pos = (robot_pos[0], robot_pos[1] + 1) 
        elif action == 'L':
            new_robot_pos = (robot_pos[0] - 1, robot_pos[1])
        elif action == 'R':
            new_robot_pos = (robot_pos[0] + 1, robot_pos[1])
        elif action == 'W':
            # Watering action
            x, y = robot_pos
            map_row, map_col = y, x  # Convert (x,y) to (row,col)
            plant_type = self.map[map_row][map_col]
            water_needed, _ = self.plant_types[plant_type]

            # Apply watering: spend water and mark plant as watered
            new_water_level = water_level - water_needed
            new_watered_plants = watered_plants | {robot_pos}

        # Check if back at origin - refill water (for movement actions)
        if action in ['U', 'D', 'L', 'R'] and new_robot_pos == (0, 0):
            new_water_level = self.water_capacity

        return (new_robot_pos, new_water_level, new_time, new_watered_plants)

    def path_cost(self, c, state1, action, state2):
        """Calculate the cost of a path from state1 to state2."""
        return c + 1  # Each action costs 1 time unit

    def goal_test(self, state):
        """Check if the goal state is reached."""
        robot_pos, water_level, time, watered_plants = state
        # Goal is reached if all plants are watered
        return len(watered_plants) == len(self.plants)   

    def solve(self):
        """Solve the problem with an uninformed search algorithm."""
        print("Initializing search...")
        solution_node = search.uniform_cost_search(self)
        return solution_node.solution() if solution_node else None
            
if __name__ == "__main__":
    # Which example to test
    example_to_test = "ex1"  # ex0, ex1, ex2, etc.
    
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
        result = plan is not None

        # Print results
        if result:
            print(f"Found a valid plan with {len(plan)} actions: {plan}")

        else:
            print("No valid plan found.")

    except Exception as e:
        print(f"Error testing {example_to_test}: {e}")