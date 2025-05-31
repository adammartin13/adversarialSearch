# Hybrid A*-Monte Carlo Adversarial Searching

The following repository contains the work of a semester-long AI search project including
each of the three game modes of each project (main, main2, main3), the agent planners
I've developed for each of them (Tom, Jerry, Spike), and the paper (LINK NEEDED) I wrote
for the final submission which you may refer to for a more thorough explanation of the final
model including formulas and sources used.

During this time I learned AI algorithms and how to implement them. The class was divided
into 32 groups to which I worked alone. By Project 2 the group agents would be pit into a
competition against one another, with my agents placing in both.

For the three projects our agents are placed in a 30x30 grid with obstacles with the goal
of capturing a target. A target is captured when both the agent and the target occupy the
same space in the same round. All agents move at the same time. In the test environment there
are 100 generated maps, with each map ran five times and agent locations randomized each game.

A winning agent is determined by the highest number of points generated after all games.
If the agent captures their target, the agent gets 3 points. If the agent runs into a
collision, the agent gets 0 points. If the agent takes too long to compute, the agent
gets 1 point.

## Project 1

Project 1 uses the main2.py (LINK NEEDED) driver code with the Tom.py (LINK NEEDED) planner
agent.

### Project 1 Rules

The agent does not compete against other teams' agents, rather simply is provided a still
target at a randomized location amongst obstacles and must find the shortest path to the target,
then capture it.

### A*

A* Search is a best-first shortest path algorithm that's used to find a goal provided a
start and end state on a grid. A* determines the value of an action via the following function:
f = g + h; where f is the total cost from start to the goal node, g is the cost accumulated
from the start to the current position, and h is a heuristic estimate of the future cost.

For our heuristic we utilize Chebyshev Distance which calculates the distance between two
points assuming that we can move in a straight line both diagonally and across. Because the
weight of moving from one place to the next in the graph is 1, the minimum number of moves
to reach a point will be exactly the Chebyshev distance.

### Results

A* search was implemented perfectly, finding the shortest path to our target as long as
the target was reachable and never crashed into obstacles.

## Project 2

Project 2 uses the main2.py (LINK NEEDED) driver code with the Jerry.py (LINK NEEDED) planner agent.

### Project 2 Rules

Our planner agent is placed into a map with two other planner agents chosen at random
from amongst the groups in the class. We must capture one of the target agents while avoiding
another whose goal is to capture us and while avoiding obstacles.

If an agent captures its target, then the agent receives 3 points and the other two receive
0. If an agent crashes into an obstacle, they receive 0 points and the other two receive 1
point. If the game times out, all agents receive 1 point.

### Adversarial A*

For our agent to incentivize decisions that promote keeping distance from the aggressor
agent we can modify our function to have penalties, with our new function being 
f = g + h + penalties.

### Monte Carlo Tree Search

In addition to A*, we modify our agent to include simulated predictive analysis with
Monte Carlo Tree Search (MCTS). MCTS is a heuristic search algorithm that combines tree
search with machine learning principles and reinforcement learning. The algorithm builds
a tree by simulating games and using their outcomes as a value for future choices.

MCTS is split into four phases: Selection, Expansion, Simulation, and Backpropagation.
In Selection we traverse the tree from its root value and select the greatest child using
a value function. For our implementation we use Upper Confidence Bound for Trees. Expansion
takes a leaf node and generates new child nodes to determine possible next moves. Simulation
simulates game rounds from a selected child node to determine potential outcomes of particular
choices. Backpropagation updates the value of child nodes based on simulation outcomes; this
net value determines which route and ultimately choice is the best decision.

Because MCTS requires some form of randomization, the entire process is simulated many times
per turn to get a wider understanding of potential outcomes. Our MCTS balances both risk
and exploration by exploiting high value moves and exploring new routes using Upper Confidence
Bound for Trees (UCT). For a breakdown of the formula, please refer to the paper (LINK NEEDED).

### Hybrid Adversarial A*-MCTS

- Results
## Project 3
- Files
- Rules of Project 3
- Final improvements to the model
- Results
- The paper
