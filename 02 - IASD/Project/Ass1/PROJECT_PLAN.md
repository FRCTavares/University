# IASD Assignment #1 - Autonomous Gardening Robot

## Project Roadmap and Task Distribution

**Team:** Francisco, Marta and Diogo  
**Start Date:** September 9, 2025  
**Deadline:** September 26, 2025  
**Deliverable:** `solution.py` implementing the `GardenerProblem` class

---

# Project Summary

## Overview

We are developing an AI system to control an autonomous gardening robot in a Martian settlement. This is essentially a **path planning and scheduling problem** where the robot must efficiently navigate a grid world to water plants before their deadlines while managing limited water resources.

## Core Problem Elements

**Environment:**

- Rectangular grid (N×M cells) representing the Martian settlement
- Cell types: obstacles (-1), empty spaces (0), plants (positive integers representing plant types)
- Robot starts at position (0,0) which is always empty and serves as the water refill station

**Plants:**

- Each plant type `k` has two critical parameters:
  - `wk`: water units required to water this plant
  - `dk`: deadline (in time steps) by which it must be watered
- Plants can only be watered **once** and **before their deadline**

**Robot Capabilities:**

- Actions: Move in 4 directions (U/D/L/R) or Water (W)
- Each action consumes exactly 1 time step
- Water capacity: W0 units (refills automatically when returning to (0,0))
- Cannot water if insufficient water remaining

**Constraints:**

- Must stay within grid boundaries
- Cannot move through obstacles
- Must respect plant deadlines
- Must have sufficient water to water each plant

## Technical Requirements

We must implement a `GardenerProblem` class inheriting from `search.Problem` with two key methods:

1. **`load(fh)`**: Parse input files with the specified format
2. **`check_solution(plan, verbose=False)`**: Validate solution plans

The solution must output a string sequence of actions (e.g., "DDWRRWUWUW").

### Input File Format

```
N M W0                    # Grid dimensions and water capacity
[N lines of grid data]    # Grid with -1=obstacle, 0=empty, k=plant type
[K lines of plant data]   # wk dk for each plant type k
```

### Output Format

- String of actions from {U, D, L, R, W}
- Example: "DDWRRWUWUW"

---

# Roadmap (September 9 → September 26, 2025)

## Week 1: Foundation & Core Implementation (Sep 9-15)

### Sep 9-10 (Mon-Tue): Project Setup & Analysis

- [ ] **Francisco, Marta, Diogo:** Set up development environment and Git repository
- [ ] **Francisco, Marta, Diogo:** Analyze sample input files and expected outputs
- [ ] **Francisco, Marta, Diogo:** Design overall architecture and data structures
- [ ] **Francisco, Marta, Diogo:** Create project skeleton with proper file structure

**Deliverables:**

- [ ] Git repository setup
- [ ] Basic project structure
- [ ] Architecture design document
- [ ] Sample input/output files for testing

### Sep 11-12 (Wed-Thu): Core Implementation Phase 1

- [ ] **Francisco:** Implement `load()` method with robust input parsing
- [ ] **Francisco:** Design internal state representation for the problem
- [ ] **Diogo:** Create basic grid visualization for debugging
- [ ] **Marta:** Start designing robot simulation framework

**Deliverables:**

- [ ] Working `load()` method
- [ ] Internal data structures defined
- [ ] Basic debugging tools

### Sep 13-15 (Fri-Sun): Core Implementation Phase 2

- [ ] **Marta:** Implement `check_solution()` method with comprehensive validation
- [ ] **Marta:** Build robot simulation engine
- [ ] **Diogo:** Create initial test cases
- [ ] **Francisco:** Refine parsing and add error handling

**Deliverables:**

- [ ] Working `check_solution()` method
- [ ] Robot simulation engine
- [ ] Initial test suite

## Week 2: Testing & Optimization (Sep 16-22)

### Sep 16-17 (Mon-Tue): Comprehensive Testing

- [ ] **Diogo:** Develop extensive test suite covering edge cases
- [ ] **Diogo:** Test with various grid sizes and plant configurations
- [ ] **Francisco, Marta, Diogo:** Validate deadline handling and water management
- [ ] **Francisco, Marta:** Bug fixes from testing

**Deliverables:**

- [ ] Comprehensive test suite
- [ ] Bug fixes and improvements
- [ ] Edge case handling

### Sep 18-19 (Wed-Thu): Code Review & Refinement

- [ ] **Francisco, Marta, Diogo:** Code review session with all team members
- [ ] **Francisco, Marta, Diogo:** Optimize performance and clean up code
- [ ] **Francisco, Marta, Diogo:** Add proper documentation and comments
- [ ] **Diogo:** Performance benchmarking

**Deliverables:**

- [ ] Reviewed and optimized code
- [ ] Complete documentation
- [ ] Performance analysis

### Sep 20-22 (Fri-Sun): Integration Testing

- [ ] **Francisco, Marta, Diogo:** End-to-end testing with complex scenarios
- [ ] **Diogo:** Stress testing with large grids and tight deadlines
- [ ] **Francisco, Marta:** Bug fixes and final optimizations
- [ ] **Francisco, Marta, Diogo:** Final integration testing

**Deliverables:**

- [ ] Fully integrated and tested solution
- [ ] Performance optimizations
- [ ] Complete bug fixes

## Week 3: Final Polish & Submission (Sep 23-26)

### Sep 23-24 (Mon-Tue): Final Testing & Documentation

- [ ] **Francisco, Marta, Diogo:** Final validation against project requirements
- [ ] **Francisco, Marta, Diogo:** Complete code documentation
- [ ] **Diogo:** Prepare submission package
- [ ] **Francisco, Marta, Diogo:** Final testing round

**Deliverables:**

- [ ] Final solution ready for submission
- [ ] Complete documentation
- [ ] Submission package prepared

### Sep 25 (Wed): Pre-submission Review

- [ ] **Francisco, Marta, Diogo:** Final team review of complete solution
- [ ] **Francisco, Marta, Diogo:** Last-minute testing and bug fixes
- [ ] **Francisco, Marta, Diogo:** Prepare for Moodle submission
- [ ] **Francisco, Marta, Diogo:** Create backup copies

**Deliverables:**

- [ ] Final reviewed solution
- [ ] Backup documentation
- [ ] Submission checklist completed

### Sep 26 (Thu): Submission Day

- [ ] **Francisco, Marta, Diogo:** Final code review and testing
- [ ] **Francisco:** Submit `solution.py` to Moodle before deadline
- [ ] **Francisco, Marta, Diogo:** Backup submission and documentation

**Deliverables:**

- [ ] `solution.py` submitted
- [ ] Complete project archive

---

# Task Distribution

## 🧑‍💻 Francisco: Input/Output & Data Structures Specialist

### Primary Responsibilities

- [ ] Implement `load(fh)` method with robust error handling
- [ ] Design and implement internal data structures (grid representation, plant database, robot state)
- [ ] Create input file parser that handles comments, blank lines, and edge cases
- [ ] Develop utility functions for grid operations and bounds checking

### Specific Tasks - Week 1 (Sep 9-15)

- [ ] **Parse grid dimensions and water capacity from first line**  
  Parse the first line of input files to extract grid dimensions (N×M) and robot's water capacity (W0). Handle whitespace and validate integer values.

- [ ] **Build 2D grid structure from input data**  
  Create a 2D list/array to represent the grid. Parse N lines after the first line, converting string input to integers. Handle obstacles (-1), empty cells (0), and plant types (k>0).

- [ ] **Create plant type dictionary with water requirements and deadlines**  
  Parse plant type definitions (K lines) after grid data. Create a dictionary mapping plant type k to tuple (wk, dk) where wk=water needed, dk=deadline.

- [ ] **Implement file reading with proper exception handling**  
  Handle file I/O with try-catch blocks. Skip comment lines (starting with #) and blank lines. Raise appropriate exceptions for malformed input.

- [ ] **Write unit tests for parsing functionality**  
  Create unit tests for load() method using pytest. Test valid inputs, edge cases (empty grids, single cell), and invalid inputs (negative dimensions, malformed files).

### Specific Tasks - Week 2 (Sep 16-22)

- [ ] **Optimize parsing and improve error handling**  
  Refactor parsing code for better performance. Add comprehensive error messages for debugging. Implement input sanitization and validation.

- [ ] **Add robust input format validation**  
  Validate grid dimensions are positive, water capacity > 0, plant types are consistent, and deadlines are reasonable. Check for missing or extra data.

- [ ] **Support code reviews of other components**  
  Review Marta's robot simulation logic and Diogo's test cases. Provide feedback on integration points and data structure usage.

- [ ] **Refine data structures based on team feedback**  
  Optimize data structures based on team feedback. Consider memory efficiency, access patterns, and ease of use for robot simulation.

### Specific Tasks - Week 3 (Sep 23-26)

- [ ] **Final testing of parsing components**  
  Conduct thorough testing of all parsing functionality. Ensure robustness against edge cases and malformed inputs.

- [ ] **Complete documentation of load() method**  
  Write comprehensive documentation including method signatures, parameter descriptions, return values, and usage examples.

- [ ] **Final integration with Marta and Diogo's components**  
  Ensure seamless integration between parsing, simulation, and testing components. Resolve any interface issues.

- [ ] **Submission preparation**  
  Prepare final code for submission, ensure clean code structure, and verify all requirements are met.

### Deliverables

- [ ] Complete and functional `load()` method
- [ ] Well-defined grid and plant data structures
- [ ] Input validation and error handling system
- [ ] Unit tests for parsing functionality

---

## 👩‍💻 Marta: Solution Validation & Robot Logic Specialist

### Primary Responsibilities

- [ ] Implement `check_solution(plan, verbose=False)` method
- [ ] Develop robot simulation engine to execute action sequences
- [ ] Create comprehensive plan validation logic
- [ ] Handle all constraint checking (bounds, obstacles, deadlines, water)

### Specific Tasks - Week 1 (Sep 9-15)

- [ ] **Design robot simulation engine architecture**  
  Design the robot simulation engine architecture. Define classes for robot state, action execution, and state transitions. Plan how to track position, water level, and time.

- [ ] **Build robot state tracking (position, water level, time)**  
  Implement RobotState class with position (x,y), water_level, and current_time attributes. Include methods to update state and check validity.

- [ ] **Implement action execution for moves (U/D/L/R) and watering (W)**  
  Create action execution methods: move_up(), move_down(), move_left(), move_right(), water_plant(). Each method updates robot state and validates the action.

- [ ] **Create basic movement validation and grid bounds checking**  
  Implement boundary checking to ensure robot stays within grid limits. Validate moves don't go through obstacles (-1 cells). Return error messages for invalid moves.

### Specific Tasks - Week 2 (Sep 16-22)

- [ ] **Complete deadline and water management validation logic**  
  Implement comprehensive deadline checking for all plants. Validate water consumption and availability. Ensure plants are watered before deadlines.

- [ ] **Validate plan against all constraints**  
  Check plans against grid boundaries, water capacity, obstacle avoidance, plant deadlines, and action validity. Provide detailed constraint violation reports.

- [ ] **Create detailed error reports for invalid plans**  
  Design comprehensive error reporting system that identifies specific constraint violations, invalid actions, and provides helpful debugging information.

- [ ] **Optimize simulation engine performance**  
  Profile and optimize simulation code for performance. Consider efficient data structures and algorithms for large grids and long action sequences.

### Specific Tasks - Week 3 (Sep 23-26)

- [ ] **Handle edge cases (empty plans, invalid actions, timing issues)**  
  Implement robust handling of edge cases including empty action sequences, invalid action characters, impossible scenarios, and timing conflicts.

- [ ] **Final testing and integration of validation system**  
  Conduct comprehensive testing of validation system with Diogo's test cases. Ensure all edge cases are properly handled.

- [ ] **Complete documentation of check_solution() method**  
  Write detailed documentation including method behavior, constraint checking logic, error handling, and usage examples.

- [ ] **Final validation with Diogo's test cases**  
  Work with Diogo to validate the system against all test cases. Ensure proper integration and consistent behavior.

### Deliverables

- [ ] Complete and robust `check_solution()` method
- [ ] Fully functional robot simulation engine
- [ ] Comprehensive constraint validation system
- [ ] Unit tests for validation functionality

---

## 🧪 Diogo: Testing & Quality Assurance Specialist

### Primary Responsibilities

- [ ] Design comprehensive test suite covering all scenarios
- [ ] Create test data files with various complexity levels
- [ ] Develop automated testing framework
- [ ] Coordinate integration testing and final validation

### Specific Tasks - Week 1 (Sep 9-15)

- [ ] **Create basic test input files (simple grids)**  
  Design and create simple test cases with small grids, few plants, and straightforward scenarios to validate basic functionality.

- [ ] **Develop test cases for complex scenarios**  
  Create complex test scenarios with large grids, multiple plant types, tight deadlines, and challenging constraint combinations.

- [ ] **Create basic automated testing framework**  
  Set up automated testing infrastructure using pytest. Create test runners and result reporting mechanisms.

- [ ] **Establish test cases for edge cases**  
  Design test cases for edge scenarios including empty grids, single cells, impossible scenarios, and boundary conditions.

### Specific Tasks - Week 2 (Sep 16-22)

- [ ] **Write unit tests for load() and check_solution() methods**  
  Create comprehensive unit tests for both main methods. Test various input scenarios, edge cases, and error conditions.

- [ ] **Develop integration tests for complete workflow**  
  Create end-to-end tests that validate the complete workflow from file loading to solution validation.

- [ ] **Create performance benchmarks and stress tests**  
  Design performance tests with large grids, many plants, and long action sequences. Establish performance baselines and identify bottlenecks.

- [ ] **Test with various grid sizes and plant configurations**  
  Create systematic tests covering different grid dimensions, plant distributions, water requirements, and deadline scenarios.

### Specific Tasks - Week 3 (Sep 23-26)

- [ ] **Final validation of all components**  
  Conduct comprehensive testing of the integrated system. Verify all components work together correctly.

- [ ] **Stress testing with large grids and tight deadlines**  
  Test system limits with very large grids, many plants, complex scenarios, and performance-critical situations.

- [ ] **Document test cases and expected behaviors**  
  Create comprehensive test documentation including test case descriptions, expected results, and coverage analysis.

- [ ] **Prepare submission package and final validation**  
  Prepare final submission package with all necessary files, documentation, and validation reports.

### Deliverables

- [ ] Comprehensive and automated test suite
- [ ] Test data files and expected outputs
- [ ] Functional automated testing framework
- [ ] Performance analysis and complete documentation

---

## 🤝 Joint Tasks (All Team Members)

### Architecture Design (Sep 9-10)

- [ ] **Francisco, Marta, Diogo:** Collaborative design of class structure and interfaces
- [ ] **Francisco, Marta, Diogo:** Agreement on coding standards and conventions
- [ ] **Francisco, Marta, Diogo:** Establish Git workflow and collaboration protocols

### Code Reviews (Sep 18-19)

- [ ] **Francisco, Marta, Diogo:** Peer review of each component
- [ ] **Francisco, Marta, Diogo:** Integration testing and joint debugging
- [ ] **Francisco, Marta, Diogo:** Performance optimization discussions

### Final Integration (Sep 23-26)

- [ ] **Francisco, Marta, Diogo:** Complete system testing
- [ ] **Francisco, Marta, Diogo:** Documentation review and completion
- [ ] **Francisco, Marta, Diogo:** Submission preparation and final validation
- [ ] **Francisco:** Official Moodle submission (designated responsible)

---

# Communication & Coordination

## Daily Workflow

- **Daily Check-ins:** Brief status updates via team chat
- **Progress Tracking:** Update shared task board with completed items
- **Issue Resolution:** Quick problem-solving sessions as needed

## Weekly Structure

- **Weekly Meetings:** In-depth progress reviews and planning sessions
- **Code Reviews:** Structured peer review sessions
- **Integration Points:** Regular testing of combined components

## Tools & Platforms

- **Code Repository:** Use private GitHub repository for version control
- **Documentation:** Maintain shared documentation for decisions and progress
- **Communication:** Team chat for daily coordination
- **Testing:** Shared testing environment and results

## Quality Assurance

- **Code Standards:** Consistent formatting and documentation
- **Testing Requirements:** Minimum test coverage for all components
- **Review Process:** All major changes require peer review
- **Final Validation:** Complete system testing before submission

---

# Success Criteria

## Technical Requirements

- ✅ Complete `GardenerProblem` class implementation
- ✅ Working `load()` method handling all input formats
- ✅ Working `check_solution()` method with comprehensive validation
- ✅ Proper error handling and edge case management
- ✅ Clean, documented, and maintainable code

## Testing Requirements

- ✅ Comprehensive test suite covering all functionality
- ✅ Edge case testing (empty grids, impossible scenarios, etc.)
- ✅ Performance testing with large inputs
- ✅ Integration testing of complete workflow

## Submission Requirements

- ✅ Single file `solution.py` submitted to Moodle
- ✅ Code passes all internal tests
- ✅ Complete and accessible documentation
- ✅ Submitted before September 26, 2025 deadline

This roadmap ensures each team member has clear, specific responsibilities while maintaining collaborative oversight for overall project success. The timeline provides adequate buffer time for testing and refinement before the September 26 deadline.

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

All solutions must pass the 10 test cases in `public1/`:

- `ex0.dat` → `ex0.plan`
- `ex1.dat` → `ex1.plan`
- ...
- `ex9.dat` → `ex9.plan`

---

# Team Coordination Specifics

## Daily Standups

- **Time**: 9:00 AM daily (15 minutes max)
- **Format**: What I did yesterday, what I'm doing today, any blockers
- **Platform**: Discord team channel
- **Backup**: Async updates if someone can't attend

## Git Workflow

- **Main branch**: Protected, requires PR review
- **Branch naming**:
  - `feature/load-method` (Francisco)
  - `feature/validation` (Marta)
  - `feature/tests` (Diogo)
- **Commit convention**: `[Component] Brief description`
  - Examples: `[Parser] Add input validation`, `[Simulation] Fix water tracking`
- **Code review**: Minimum 1 approval before merge
- **Integration**: Daily merges to avoid conflicts

## Communication Channels

- **Daily coordination**: Discord #iasd-project channel
- **Code discussions**: GitHub PR comments and issues
- **Emergency contact**: WhatsApp group
- **File sharing**: GitHub repository + Google Drive backup
- **Documentation**: GitHub Wiki for architecture decisions

## Meeting Schedule

- **Weekly sync**: Mondays 2:00 PM (1 hour)
- **Code review sessions**: Wednesdays 4:00 PM (30 minutes)
- **Integration testing**: Fridays 3:00 PM (45 minutes)
- **Emergency sessions**: As needed, minimum 2 hours notice

---

# Enhanced Testing Strategy

## Test Categories

### 1. Unit Tests (Each person tests their components)

**Francisco - Parser Tests:**

- Valid input parsing
- Malformed file handling
- Comment and blank line skipping
- Edge cases: empty grids, single cell, negative values
- Performance: files up to 100KB

**Marta - Validation Tests:**

- Robot movement in all directions
- Water consumption tracking
- Deadline enforcement
- Obstacle collision detection
- Boundary checking
- Integration with Francisco's data structures

**Diogo - System Tests:**

- End-to-end workflow testing
- Performance benchmarks
- Stress testing with large scenarios
- Validation against project requirements

### 2. Integration Tests Using public1/ Data

- **Acceptance Criteria**: All 10 test cases must pass
- **Test Files**: ex0.dat through ex9.dat with corresponding .plan files
- **Automated Testing**: Run all tests on every PR
- **Performance Target**: Complete test suite in < 30 seconds

### 3. Performance Requirements

- **load() method**: < 1 second for files up to 100KB
- **check_solution()**: < 5 seconds for action sequences up to 10,000 steps
- **Memory usage**: Efficient for grids up to 1000×1000
- **Test coverage**: Minimum 90% code coverage

## Test Automation Framework

```python
# Example test structure
def test_load_basic_grid():
    problem = GardenerProblem()
    with open('ex0.dat', 'r') as fh:
        problem.load(fh)
    assert problem.grid_size == (3, 4)
    assert problem.water_capacity == 100

def test_check_solution_valid():
    problem = GardenerProblem()
    # Load test data
    result = problem.check_solution("DDWRRWUWUW")
    assert result == True
```

---

# Risk Management & Contingencies

## Technical Risks

### Integration Issues

- **Prevention**: Weekly integration checkpoints
- **Detection**: Automated integration tests on every merge
- **Mitigation**: Clear component interfaces, mock objects for testing

### Performance Problems

- **Prevention**: Profile early with realistic data sizes
- **Detection**: Automated performance tests
- **Mitigation**: Optimize critical paths, consider algorithm improvements

### Complex Edge Cases

- **Prevention**: Start with simple tests, gradually increase complexity
- **Detection**: Comprehensive test suite including boundary conditions
- **Mitigation**: Systematic debugging, team problem-solving sessions

## Team Risks

### Member Unavailability

- **Prevention**: Cross-training on critical components
- **Detection**: Daily check-ins and progress tracking
- **Mitigation**: Detailed documentation, pair programming sessions

### Deadline Pressure

- **Prevention**: Daily progress tracking with early warning system
- **Detection**: Weekly milestone reviews
- **Mitigation**: Scope reduction, parallel development, extended work sessions

### Merge Conflicts

- **Prevention**: Clear component boundaries, frequent small merges
- **Detection**: Git conflict detection
- **Mitigation**: Dedicated merge conflict resolution sessions

## Contingency Plans

### Week 1 Delays

- **Action**: Reduce optimization scope, focus on core functionality
- **Responsibility**: All team members work in parallel on critical path items
- **Escalation**: Daily extended sessions if > 1 day behind

### Week 2 Delays

- **Action**: Parallel development, reduce documentation scope
- **Responsibility**: Focus on essential features only
- **Escalation**: Weekend work sessions, external help if needed

### Week 3 Critical Issues

- **Action**: Emergency team sessions, simplified submission
- **Responsibility**: All hands on deck, cancel other commitments
- **Escalation**: Submit minimal viable solution if necessary

---

# Tools & Environment Setup

## Development Environment

- **IDE**: VS Code with Python extension pack
- **Python Version**: 3.8+ (check compatibility with search module)
- **Virtual Environment**: Use `venv` or `conda` for dependencies
- **Code Formatter**: Black for consistent formatting
- **Linter**: flake8 for style checking

## Version Control Setup

```bash
# Repository structure
IASD-Project/
├── solution.py          # Final submission file
├── src/                 # Development modules
│   ├── parser.py       # Francisco's parsing logic
│   ├── validator.py    # Marta's validation logic
│   └── utils.py        # Shared utilities
├── tests/              # Test files
│   ├── test_parser.py  # Francisco's tests
│   ├── test_validator.py # Marta's tests
│   └── test_integration.py # Diogo's tests
├── data/               # Test data
│   └── public1/        # Official test cases
└── docs/               # Documentation
```

## Testing Infrastructure

- **Framework**: pytest for test execution
- **Coverage**: pytest-cov for coverage measurement
- **CI/CD**: GitHub Actions for automated testing
- **Performance**: pytest-benchmark for performance tests

## Communication Tools

- **Chat**: Discord server with dedicated channels
- **Video**: Zoom for meetings and pair programming
- **Documentation**: GitHub Wiki + Google Docs for collaborative editing
- **Task Tracking**: GitHub Projects with Kanban board

---

# Final Submission Checklist

## Code Quality Requirements

- [ ] Single file `solution.py` contains complete `GardenerProblem` class
- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have docstrings with clear descriptions
- [ ] No external dependencies beyond standard library and `search` module
- [ ] Code is clean, readable, and maintainable

## Testing Requirements

- [ ] All 10 public test cases pass (ex0.dat through ex9.dat)
- [ ] Unit tests achieve >90% code coverage
- [ ] Performance requirements met (load <1s, check_solution <5s)
- [ ] Edge cases handled properly (empty plans, invalid actions, etc.)
- [ ] Integration tests pass for complete workflow

## Documentation Requirements

- [ ] README.md with usage instructions
- [ ] Code comments explain complex algorithms
- [ ] Architecture decisions documented
- [ ] Test case descriptions and expected behaviors
- [ ] Performance analysis and optimization notes

## Submission Process

- [ ] Final code review by all team members
- [ ] Backup copies created in multiple locations
- [ ] Submission file tested on clean environment
- [ ] Moodle submission completed before deadline
- [ ] Confirmation email received and verified

This enhanced roadmap provides the missing technical details, coordination specifics, and risk management that will ensure your team's success!
