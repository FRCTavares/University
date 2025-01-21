The goal of swarm intelligence is to design intelligent multi-agent systems by taking inspiration from the collective behavior of social insects such as ants, termites, bees, wasps, and other animal societies such as flocks of birds or schools of fish.

### Historical Background
ACO is inspired by the foraging behavior of ant colonies. Ants prefer community survival rather than individuality. They communicate mainly through pheromones, which are organic chemical compounds secreted by ants, triggering a social response in other ants. These compounds act like hormones outside the body. Ants leave pheromone trails in the soil for other ants to smell.
The underlying principle of ACO is observing the ants' movements from their nests to find food in the shortest possible distance. They initially move randomly around the nest, opening multiple routes from the nest to the food. Based on the quantity and quality of the food, ants carry a portion of it back, leaving a necessary pheromone concentration on their return. Depending on the pheromone trail, the path to the food is selected based on the concentration as well as the rate of evaporation of the pheromone. With this last deciding feature, the length of the path will likely be the shortest and is therefore accounted for indirectly.

### Simple Example of How the Algorithm Works
Let's consider a diagram where there is a colony, food, and only two paths: one direct and the other curved, therefore obviously longer.

- All the ants start in the nest. There are 6 in total. There is no pheromone content anywhere, so the ants begin the search for food with equal distribution probability. Thus, 3 go through the curved path and the other 3 through the direct trail. Naturally, the ants that took the latter path arrive at the food source first.
- Now these ants face the same selection dilemma: "Should they go through the original trail or through the new path (the curved one)?" But the difference now is that pheromone traces already exist in the original trail, so the probability of choosing that path is greater.
- Evidently, more ants choose the original path, enriching it further with pheromones. Therefore, the whole colony gradually uses the shorter path, optimizing the route.

### Algorithm Design
Considering the previous behaviour of the ants we can design a mathmatical algorithm mimicking it.
To simplify the explanation only a single food source and single ant colony have been considered with just two possible paths. The ant colony and the food are vertices of the graph and the trails are edges, and the pheromone values are theire weight.
The graph is G = (V,E), we have V_s (Ant Colony) and V_d (Fod Source), E_1 and E_2 with lenghts L_1 and L_2, and finnaly the pheromone values can be R_1 and R_2 for teh corresponding vertices.
The starting probability of selection of path (E_1 or E_2) can be expressed as:

P_i = frac{R_i}{R_1 + R_2}; i = 1, 2

While returning through this shortest path say E_i, the pheromone value is updated for the corresponding path. This step is done based on the lenght of the paths as well as the evaporation rate of the pheronome.

Update can be step-wise realized as follows:

1. In accordance to path lenght: R_i <- R_i + frac{K}{L_i}

K serves as a parameter of the model.

2. In accordance to evaporation rate: R_i <- (1 - v)*R_i

Parameter v belongs to [0,1] that regulate the pheronome evaporation

### Pseudocode:

Procedure AntColonyOptimization:
    Initialize necessary parameters and pheromone trials;
    while not termination do:
        Generate ant population;
        Calculate fitness values associated with each ant;
        Find best solution through selection methods;
        Update pheromone trial;
    end while
end procedure



