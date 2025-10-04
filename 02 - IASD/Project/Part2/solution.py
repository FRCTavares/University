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
        self.visited_states = {}  # Dictionary to track (pos, water, watered_plants) -> best_time
        
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
        possible_actions = self.actions(state)

        if action not in possible_actions:
            return state  # Invalid action, return the same state
        
        robot_pos, water_level, time, watered_plants = state
        
        # Initialize new state variables
        new_robot_pos = robot_pos
        new_water_level = water_level
        new_watered_plants = watered_plants

        if action in possible_actions:
            if action == 'U' or action == 'D' or action == 'L' or action == 'R':
                # Movement actions
                x, y = robot_pos

                if action == 'U':
                    new_robot_pos = (x, y - 1)
                elif action == 'D':
                    new_robot_pos = (x, y + 1) 
                elif action == 'L':
                    new_robot_pos = (x - 1, y)
                elif action == 'R':
                    new_robot_pos = (x + 1, y)

                # Check if back at origin - refill water
                if new_robot_pos == (0, 0):
                    new_water_level = self.water_capacity
        
            elif action == 'W':
                # Watering action
                # Find plant type at current position to get water requirement
                x, y = robot_pos
                map_row, map_col = y, x  # Convert (x,y) to (row,col)
                plant_type = self.map[map_row][map_col] # type: ignore
                water_needed, _ = self.plant_types[plant_type]

                # Apply watering: spend water and mark plant as watered
                new_water_level = water_level - water_needed
                new_watered_plants = watered_plants | {robot_pos}

            # Increment time
            new_time = time + 1
                    
            # Dominance check: if we've been in this situation before with less or equal time, discard
            state_key = (new_robot_pos, new_water_level, frozenset(new_watered_plants))
            
            if state_key in self.visited_states:
                if self.visited_states[state_key] <= new_time:
                    # We've been here before in less or equal time, discard this successor
                    return None
                else:
                    # This is better, update the best time for this state
                    self.visited_states[state_key] = new_time
            else:
                # First time visiting this state
                self.visited_states[state_key] = new_time

        return (new_robot_pos, new_water_level, new_time, new_watered_plants)

    def path_cost(self, c, state1, action, state2):
        """Calculate the cost of a path from state1 to state2."""
        return super().path_cost(c, state1, action, state2) + 1  # Each action costs 1 time unit

    def goal_test(self, state):
        """Check if the goal state is reached."""
        # Not used in this context
        pass   

    def solve(self):
        """Solve the problem."""
        # Not used in this context
        pass

    def check_solution(self, solution):
        """
        Simulate the solution step by step and validate all constraints.
        
        Args:
            solution: string of actions like "URDLW"
        
        Returns:
            True if valid, False otherwise
        """
        # Validate solution contains only valid characters
        valid_actions = {'U', 'D', 'L', 'R', 'W'}
        for char in solution:
            if char not in valid_actions:
                return False
            
        # Initialize simulation state
        robot_pos = (0, 0)  # Start at origin
        time = 0
        water = self.water_capacity
        watered_plants = set()
        
        # Get map dimensions
        if not self.map:
            return False
        
        height = len(self.map)
        width = len(self.map[0]) if height > 0 else 0
    
        # Simulate each action: It will cycle trough every action in soltuion file
        for i, action in enumerate(solution):

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
                    return False
                
                # Check obstacles
                if new_pos in self.obstacles:
                    return False
                
                # Valid move
                robot_pos = new_pos

                # Check if back at origin - refill water
                if robot_pos == (0, 0):
                    water = self.water_capacity

            elif action == 'W':
                # Watering action
            
                # Check if there's a plant at current position
                if robot_pos not in self.plants:
                    return False
                
                # Check if plant already watered
                if robot_pos in watered_plants:
                    return False
                
                # Get plant info
                plant_deadline = self.plants[robot_pos]
                
                # Find plant type to get water requirement
                plant_type = None
                map_row, map_col = robot_pos[1], robot_pos[0]  # Convert (x,y) to (row,col)
                if 0 <= map_row < len(self.map) and 0 <= map_col < len(self.map[map_row]):
                    plant_type = self.map[map_row][map_col]
                
                if plant_type is None or plant_type not in self.plant_types:
                    return False
                
                water_needed, _ = self.plant_types[plant_type]
            
                # Check if enough water
                if water < water_needed:
                    return False
                
                # Check deadline
                if time > plant_deadline:
                    return False
                
                # Water the plant
                watered_plants.add(robot_pos)
                water -= water_needed

            time += 1  # Each action takes 1 time unit

        # Final validation: all plants must be watered exactly once
        if len(watered_plants) != len(self.plants):
            return False
        
        return True
            
if __name__ == "__main__":
    # Which example to test
    example_to_test = "ex0"  # ex0, ex1, ex2, etc.
    
    # Test the specified example
    dat_file = Path(f"tests/{example_to_test}.dat")
    
    if not dat_file.exists():
        print(f"File {dat_file} not found!")
        exit(1)
    
    print(f" Solving for {example_to_test}")
    print("=" * 50)
    
    try:
        # Create problem instance
        problem = GardenerProblem()
        
        # Load the problem with file handle
        with open(dat_file, 'r') as f:
            problem.load(f)
        
        """
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
        """
        
        possible_actions = problem.actions(((0,0), problem.water_capacity, 0, frozenset()))
        print(possible_actions)

    except Exception as e:
        print(f"Error testing {example_to_test}: {e}")