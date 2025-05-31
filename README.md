# Hybrid A*-Monte Carlo Adversarial Searching

The following repository contains the work of a semester-long AI search project including
each of the three game modes of each project (main, main2, main3), the agent planners
I've developed for each of them (Tom, Jerry, Spike), and the [paper](https://github.com/adammartin13/adversarialSearch/blob/main/Paper.pdf)
I wrote for the final submission.

The class was divided
into 32 groups to which I worked alone. By Project 2 the group agents would be pit into a
competition against one another, with my agents placing in both.

For the three projects our agents are placed in a 30x30 grid with obstacles and the goal
of capturing a target. A target is captured when both the agent and the target occupy the
same space in the same round. All agents move at the same time. In the test environment there
are 100 generated maps, with each map ran five times and agent locations randomized each game.

A winning agent is determined by the highest number of points generated after all games.
If the agent captures their target, the agent gets 3 points. If the agent runs into a
collision, the agent gets 0 points. If the agent takes too long to compute, the agent
gets 1 point.

## Project 1

Project 1 uses the [main2.py](https://github.com/adammartin13/adversarialSearch/blob/main/main2.py) driver code with the 
[Tom.py](https://github.com/adammartin13/adversarialSearch/blob/main/planners/tom.py) planner agent.

### Project 1 Rules

The agent does not compete against other teams' agents, rather is simply provided a still
target at a randomized location amongst obstacles and must find the shortest path to the target.

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

Project 2 uses the [main3.py](https://github.com/adammartin13/adversarialSearch/blob/main/main3.py) 
driver code with the [Jerry.py](https://github.com/adammartin13/adversarialSearch/blob/main/planners/jerry.py) planner agent.

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
Bound for Trees (UCT). For a breakdown of the formula, please refer to the [paper](https://github.com/adammartin13/adversarialSearch/blob/main/Paper.pdf).

### Hybrid Adversarial A*-MCTS

A* is fast and will reliably return the shortest path to our target, however MCTS allows us to
simulate many possible outcomes to determine the greatest long-term strategy at the cost of being
much more computationally expensive. To get the best of both, our agent relies on A* unless within
a short specified range to either the target or the aggressor.

### Results

Our agents MCTS implementation was rather flawed, unable to take full advantage of simulation rollouts.
Despite this flaw, the agent managed to place in the first competition across the class.

## Project 3

Project 3 uses the [main.py](https://github.com/adammartin13/adversarialSearch/blob/main/main.py) driver and the [Spike.py](https://github.com/adammartin13/adversarialSearch/blob/main/planners/spike.py)
search agent. [devel.py](https://github.com/adammartin13/adversarialSearch/blob/main/devel.py) is a test file included for testing agents.

### Project 3 Rules

Our agents now have set probabilities, unknown to them and randomized per game, where a desired returned action has a
percent chance of being left or right shifted by 90 degrees. To prevent collisions the agents must now perform estimations
and properly utilize them in their desired outcomes.

In addition to modifying our agent to operate under the new game conditions, we also added a series of action overrides
to metagame the conditions of our environment.

### Bayesian Estimation with Dirichlet Prior

Our probability estimator combines Maximum Likelihood Estimation with Bayesian Estimation and Laplace smoothing 
where we use previous knowledge to increase our likelihood of correctly estimating probabilities as our agent plays
the game. The probability of an action occuring is equal to the number of times the actions result has been observed
divided by the total number of actions recorded.

### Crash

The Crash override allows us to intentionally run our agent into an obstacle in the event that either the target
cannot be captured of if the next action would risk capture by the agent pursuing ours and there's an obstacle within
range. This allows our agent to have a lower loss point ratio relative to surrending a win to the opposing agents.

### Dodge

The Dodge override allows us to take advantage of the lack of in-game physics by allowing our agent to phase through and
swap positions with the agent pursuing us. This assumes that the opposing agent will choose to not stand still. If succesful,
this not only helps us avoid capture but potentially puts us in a path towards our intended target.

### Slide

The Slide override allows us to avoid crashing when adjacent to obstacles by finding the heuristically most favorable outcome
rotation probabilities in mind.

### Results

Our agent in this final implementation not only patched the rollout simulation errors of the previous project, but also effectively
determines probabilities and utilizes the series of overrides we've implemented. We placed in the competition against other agents
and received a perfect score in the class.

### The Paper
For a more thorough breakdown on the final implementation with formulas and sources, please refer to the [paper](https://github.com/adammartin13/adversarialSearch/blob/main/Paper.pdf).
