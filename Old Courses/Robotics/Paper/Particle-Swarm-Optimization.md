PSO is inspired once again in the way animals (Swarms, Schools, flocks...) collective make decisions based on their own experience and their colective knowladge.

### Key Features

 - Each robot adjusts its trajectory based on its own experience and the swarm's collective knowledge.
 - Swarm robotics use this to optimize task allocation, formation control, and exploration.

Basically this a technique fundamental to swarm robotics beacuse it decentralize control, enhances cooperations and adaprability in th swarm.

###

Optimization is the process of finding optimal values for some parameters to fulfill all design requirements while keeping the cost as low as possible.

The conventional optimization algorithms tend to be deterministic which makes them fail often since they have single-base solutions, converge to local minimums instead of global, and a lot of times the search space has unkown spaces.

### Particle Swarm Optimization

Particle Swarm Optimization (PSO) is a meta-heuristic optimization algorithm inspired by the collective behavior observed in schools of fish, flocks of birds, and other natural systems. It was originally developed to simulate the elegant yet unpredictable movement of bird flocks and has since been adapted for solving complex optimization problems.

### Key Concepts:
 - Swarm Behavior: Individuals in a swarm (particles) have limited observable ranges. However, by leveraging the collective behavior of the group, a larger solution space can be explored effectively.

 - Social System Simulation: PSO mimics a simplified social system where particles share information and adapt based on their own experiences and those of their neighbors.

### Model

1. Particles & Attributes:
    - Each particle represents a candidate solution in the serach space.
    - Each particle has:
        A position vector.
        A velocity vector.
        A fitness value determined by the objetive function.
2. Memory and Bes Solutions:
    - Each particle tracks its:
        Personal Best, which corresponds to the best postion and fitness value it has achieved so far.
    - The swarm tracks the:
        Global Best, corresponding to the best position and fitness value achieved by any particle in the swarm.
3. Position and Velocity Updates:
    - At each iteration, a particle updates its velocity and position based on:
        Its current velocity.
        The difference between its position and its Personal Best.
        And The difference between its position and the Global Best.

The update equations are:

v_i(t+1) = w * v_i(t) + c_1*r_1*(P_best,i - x_i(t)) +c_2*r_2*(G_best - x_i(t))

x_i(t+1) = x_i(t) + v_i(t+1)

where w is the inertia weight corresponding to the balance between exploration and exploitaion, c_1 and c_2 are the acceleration coefficients, and this hyperparameters and previously defined.
r_1 and r_2 are random numbers betweem 0 and 1. 
v_i and x_i are the velocity and position fo particle i.


### PSO Algorithm 

Parameters of problem:
- Number of dimensions (d)
- Lower bound (minx)
- Upper bound (maxx)

Hyperparameters of the algorithm:  
- Number of particles (N)
- Maximum number of iterations (max_iter)
- Inertia (w)
- Cognition of particle (C1)
- Social influence of swarm (C2)

Step1: Randomly initialize Swarm population of N particles Xi ( i=1, 2, …, n)

Step2: Select hyperparameter values
           w, c1 and c2
           
Step 3: For Iter in range(max_iter):  # loop max_iter times  
            For i in range(N):  # for each particle:
               a. Compute new velocity of ith particle
                    swarm[i].velocity = 
                         w*swarm[i].velocity + 
                         r1*c1*(swarm[i].bestPos - swarm[i].position) +
                         r2*c2*( best_pos_swarm - swarm[i].position) 
               b. Compute new position of ith particle using its new velocity
                    swarm[i].position += swarm[i].velocity
               c. If position is not in range [minx, maxx] then clip it
                    if swarm[i].position < minx:
                        swarm[i].position = minx
                    elif swarm[i].position > maxx:
                        swarm[i].position = maxx
               d. Update new best of this particle and new best of Swarm
                     if swaInsensitive to scaling of design variables.rm[i].fitness < swarm[i].bestFitness:
                        swarm[i].bestFitness = swarm[i].fitness
                        swarm[i].bestPos = swarm[i].position

                     if swarm[i].fitness < best_fitness_swarm
                        best_fitness_swarm = swarm[i].fitness
                        best_pos_swarm = swarm[i].position
             End-for
         End -for
Step 4: Return best particle of Swarm


This algorithm is really good beacause it is insensitive to scaling of design variables, derivative free, very few algorithm parameters, very enffient at global search, and it is easily parallelized for concurrent processing.

Its main disadvanetage is its slow convergence in the refined search stage.