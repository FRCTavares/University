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

        print(f"Loading problem from file handle")
        
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
        
        print(f"Grid size: {N}x{M}, Water capacity: {W0}")
        
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
        
        print(f"Plant types: {self.plant_types}")
        
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
        Placeholder for solution checking - will be implemented later.
        For now, just return True to test file loading.
        """
        print(f"Checking solution: {solution[:20]}..." if len(solution) > 20 else f"Checking solution: {solution}")
        return True  # Placeholder - always return True for now
if __name__ == "__main__":
    # Specify which example to test here
    example_to_test = "ex0"  # Change this to test different examples: ex0, ex1, ex2, etc.
    
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
        result = problem.check_solution(plan)
        
        # Print summary
        print(f"\n Summary for {example_to_test}:")
        if problem.map is not None:
            print(f"   Grid: {len(problem.map)}x{len(problem.map[0]) if problem.map and len(problem.map) > 0 else 0}")
        else:
            print("   Grid: 0x0")
        print(f"   Water capacity: {problem.water_capacity}")
        print(f"   Plants: {len(problem.plants)}")
        print(f"   Obstacles: {len(problem.obstacles)}")
        print(f"   Plan length: {len(plan)} actions")
        print(f"   Result: {'VALID' if result else 'INVALID'}")
        
    except Exception as e:
        print(f"Error testing {example_to_test}: {e}")