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
from typing import Dict, List

# Define the problem class
class GardenerProblem(search.Problem):
    def __init__(self, initial_state, goal=None):
        super().__init__(initial_state, goal)  # Initialization of the base class with the initial state and goal state.
        
    def load(self, filename):
        print(f"Loading problem from {filename}")
        with open(filename, 'r') as file:
            data = file.read()

        sections = self.parse_file(filename)
        # Set initial robot position and map from parsed sections
        initial_state = {
            "robot": (0, 0),
            "map": sections.get("The map")
        }
        self.initial_state = initial_state
        print(f"Initial state loaded: {self.initial_state}")

    def parse_file(path: str | Path) -> Dict[str, List[List[int]]]:
        """
        Parse a file into sections:
        - Lines starting with '#' open a new section (key = cleaned text after '#').
        - Following numeric lines are stored under that section until the next '#'.
        - Returns a dict {section_name: list of numeric lines}.
        """
        path = Path(path)
        sections: Dict[str, List[List[int]]] = {}
        current_key = None

        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue  # skip blanks

                if line.startswith("#"):
                    # Section name without '#'
                    current_key = line.lstrip("#").strip()
                    sections[current_key] = []
                    continue

                if current_key is not None:
                    tokens = line.split()
                    try:
                        nums = [int(t) for t in tokens]
                        sections[current_key].append(nums)
                    except ValueError:
                        # Skip lines that aren't fully numeric
                        pass

        return sections

    def check_solution(self, state):
        print(f"Checking solution for state: {state}")
        if state is None:
            return False
        elif state.is_goal():
            return True
        else:
            return False