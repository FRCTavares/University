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
        self.initial_state = None
        self.map = None
        self.water_capacity = None
        self.plants = {}
        self.obstacles = set()
        self.plant_types = {}  # Maps plant type number to (water_needed, deadline)
        
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
        
        #print(f"Grid size: {N}x{M}, Water capacity: {W0}")
        
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
        
        #print(f"Plant types: {self.plant_types}")
        
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

    def check_solution(self, solution, verbose=False):
        """
        Simulate the solution step by step and validate all constraints.
        
        Args:
            solution: string of actions like "URDLW"
            verbose: if True, print diagnostic information
        
        Returns:
            True if valid, False otherwise
        """
        if verbose:
            print(f"Starting solution validation for plan: {solution}")

        # Validate solution contains only valid characters
        valid_actions = {'U', 'D', 'L', 'R', 'W'}
        for char in solution:
            if char not in valid_actions:
                if verbose:
                    print(f"Invalid action character: {char}")
                return False
            
        # Initialize simulation state
        robot_pos = (0, 0)  # Start at origin
        time = 0
        water = self.water_capacity
        watered_plants = set()
        
        # Get map dimensions
        if not self.map:
            if verbose:
                print("No map loaded")
            return False
        
        height = len(self.map)
        width = len(self.map[0]) if height > 0 else 0

        if verbose:
            print(f"Initial state: pos={robot_pos}, water={water}, time={time}")
    
        # Simulate each action: It will cycle trough every action in soltuion file
        for i, action in enumerate(solution):
            time += 1  # Each action takes 1 time unit
            
            if verbose:
                print(f"Step {i+1}: Action '{action}' at time {time}")
            
            if action in 'UDLR':
                # Movement actions
                x, y = robot_pos

                if action == 'U':
                    new_pos = (x, y - 1) # Move up decreases y
                elif action == 'D':
                    new_pos = (x, y + 1) # Move down increases y
                elif action == 'L':
                    new_pos = (x - 1, y) # Move left decreases x
                elif action == 'R':
                    new_pos = (x + 1, y) # Move right increases x

                new_x, new_y = new_pos

                # Check bounds
                if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
                    if verbose:
                        print(f"Movement out of bounds: {robot_pos} -> {new_pos}")
                    return False
                
                # Check obstacles
                if new_pos in self.obstacles:
                    if verbose:
                        print(f"Movement into obstacle at {new_pos}")
                    return False
                
                # Valid move
                robot_pos = new_pos

                # Check if back at origin - refill water
                if robot_pos == (0, 0):
                    water = self.water_capacity
                    if verbose:
                        print(f"Refilled water at origin, water={water}")
                
                if verbose:
                    print(f"Moved to {robot_pos}, water={water}")

            elif action == 'W':
                # Watering action
            
                # Check if there's a plant at current position
                if robot_pos not in self.plants:
                    if verbose:
                        print(f"Tried to water at {robot_pos} but no plant there")
                    return False
                
                # Check if plant already watered
                if robot_pos in watered_plants:
                    if verbose:
                        print(f"Plant at {robot_pos} already watered")
                    return False
                
                # Get plant info
                plant_deadline = self.plants[robot_pos]
                
                # Find plant type to get water requirement
                plant_type = None
                map_row, map_col = robot_pos[1], robot_pos[0]  # Convert (x,y) to (row,col)
                if 0 <= map_row < len(self.map) and 0 <= map_col < len(self.map[map_row]):
                    plant_type = self.map[map_row][map_col]
                
                if plant_type is None or plant_type not in self.plant_types:
                    if verbose:
                        print(f"Invalid plant type at {robot_pos}")
                    return False
                
                water_needed, _ = self.plant_types[plant_type]
            
                # Check if enough water
                if water < water_needed:
                    if verbose:
                        print(f"Not enough water: need {water_needed}, have {water}")
                    return False
                
                # Check deadline
                if time > plant_deadline:
                    if verbose:
                        print(f"Missed deadline for plant at {robot_pos}: time {time} > deadline {plant_deadline}")
                    return False
                
                # Water the plant
                watered_plants.add(robot_pos)
                water -= water_needed

                if verbose:
                    print(f"Watered plant at {robot_pos}, used {water_needed} water, remaining={water}")
    
        # Final validation: all plants must be watered exactly once
        if len(watered_plants) != len(self.plants):
            if verbose:
                unwatered = set(self.plants.keys()) - watered_plants
                print(f"Not all plants watered. Missing: {unwatered}")
            return False
        
        if verbose:
            print(f"Solution valid! All {len(self.plants)} plants watered within deadlines")
        
        return True
            
if __name__ == "__main__":
    # Which example to test
    example_to_test = "ex0"  # ex0, ex1, ex2, etc.
    
    # Test the specified example
    dat_file = Path(f"public1/{example_to_test}.dat")
    plan_file = Path(f"public1/{example_to_test}.plan")
    
    if not dat_file.exists():
        print(f"File {dat_file} not found!")
        exit(1)
    
    if not plan_file.exists():
        print(f"Plan file {plan_file} not found!")
        exit(1)
    
    print(f" Testing {example_to_test}")
    print("=" * 50)
    
    try:
        # Create problem instance
        problem = GardenerProblem()
        
        # Load the problem with file handle
        with open(dat_file, 'r') as f:
            problem.load(f)
        
        # Read the solution
        with open(plan_file, 'r') as f:
            plan = f.read().strip()
        
        # Test the solution
        result = problem.check_solution(plan, verbose=True)
        
        # Print summary
        print(f"\n Summary for {example_to_test}:")
        if problem.map is not None:
            print(f"   Grid: {len(problem.map)}x{len(problem.map[0]) if problem.map and len(problem.map) > 0 else 0}")
        else:
            print("   Grid: 0x0")
        print(f"   Water capacity: {problem.water_capacity}")
        print(f"   Plants: {len(problem.plants)}")
        print(f"   Obstacles: {len(problem.obstacles)}")
        print(f"   Plan length: {len(plan)} actions; Plan: {plan}")
        print(f"   Result: {'VALID' if result else 'INVALID'}")
        
    except Exception as e:
        print(f"Error testing {example_to_test}: {e}")