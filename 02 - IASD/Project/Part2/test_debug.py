import search
from pathlib import Path

class GardenerProblem(search.Problem):
    def __init__(self):
        self.map = []
        self.water_capacity = None
        self.plants = {}
        self.obstacles = set()
        self.plant_types = {}
        self.best_state = {}
        self.expanded_count = 0
        
    def load(self, fh):
        lines = []
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        
        if not lines:
            raise ValueError("No valid lines found in file")
        
        first_line = lines[0].split()
        N, M, W0 = int(first_line[0]), int(first_line[1]), int(first_line[2])
        self.water_capacity = W0
        
        self.map = []
        for i in range(1, N + 1):
            row = [int(x) for x in lines[i].split()]
            self.map.append(row)
        
        K = 0
        for row in self.map:
            for cell in row:
                if cell > 0:
                    K = max(K, cell)
        
        plant_def_start = 1 + N
        self.plant_types = {}
        for i in range(K):
            parts = lines[plant_def_start + i].split()
            wk, dk = int(parts[0]), int(parts[1])
            self.plant_types[i + 1] = (wk, dk)
        
        self.plants = {}
        self.obstacles = set()
        
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                cell_value = self.map[row][col]
                pos = (col, row)
                if cell_value == -1:
                    self.obstacles.add((col, row))
                elif cell_value > 0:
                    wk, dk = self.plant_types[cell_value]
                    self.plants[pos] = (cell_value, wk, dk)
        
        initial_state = ((0, 0), self.water_capacity, 0, frozenset())
        super().__init__(initial_state, None)
        self.best_state = {}
    
    def actions(self, state):
        self.expanded_count += 1
        if self.expanded_count % 1000 == 0:
            print(f"Expanded {self.expanded_count} states, best_state size: {len(self.best_state)}")
        
        pos, water, time, watered = state
        
        # NO DOMINANCE PRUNING
        
        x, y = pos
        height = len(self.map)
        width = len(self.map[0]) if height > 0 else 0
        possible_actions = []
        
        moves = {
            'L': (x - 1, y),
            'R': (x + 1, y),
            'U': (x, y - 1),
            'D': (x, y + 1)
        }
        
        for action, (nx, ny) in moves.items():
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in self.obstacles:
                possible_actions.append(action)
        
        if pos in self.plants and pos not in watered:
            _, wk, dk = self.plants[pos]
            if water >= wk and time + 1 <= dk:
                possible_actions.append('W')
        
        return possible_actions
    
    def result(self, state, action):
        robot_pos, water_level, time, watered_plants = state
        height = len(self.map)
        width = len(self.map[0]) if height > 0 else 0
        
        new_robot_pos = robot_pos
        new_water_level = water_level
        new_watered_plants = watered_plants
        x, y = robot_pos
        
        if action == 'L':
            new_robot_pos = (x - 1, y)
        elif action == 'R':
            new_robot_pos = (x + 1, y)
        elif action == 'U':
            new_robot_pos = (x, y - 1)
        elif action == 'D':
            new_robot_pos = (x, y + 1)
        elif action == 'W':
            _, wk, dk = self.plants[robot_pos]
            new_water_level = water_level - wk
            new_watered_plants = watered_plants | {robot_pos}
        
        if action in ['L', 'R', 'U', 'D']:
            if new_robot_pos == (0, 0):
                new_water_level = self.water_capacity
        
        new_time = time + 1
        
        return (new_robot_pos, new_water_level, new_time, new_watered_plants)
    
    def path_cost(self, c, state1, action, state2):
        return c + 1
    
    def goal_test(self, state):
        _, _, _, watered = state
        return len(watered) == len(self.plants)
    
    def solve(self):
        print("Initializing search (NO PRUNING - will take longer)...")
        solution_node = search.uniform_cost_search(self)
        
        if solution_node:
            return ''.join(solution_node.solution())
        return None

# Test
problem = GardenerProblem()
with open('public2/ex1.dat', 'r') as f:
    problem.load(f)

print(f"Plants: {problem.plants}")
print(f"Water capacity: {problem.water_capacity}")
print(f"Grid: {len(problem.map)}x{len(problem.map[0])}")
print()

plan = problem.solve()
if plan:
    print(f"\nFound plan (len={len(plan)}): {plan}")
    print(f"Total states expanded: {problem.expanded_count}")
else:
    print(f"\nNo plan found after {problem.expanded_count} expansions")
