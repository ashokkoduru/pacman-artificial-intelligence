# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    # Depth First Search uses a stack

    visited = []  # Our stack
    checked = []
    present_state = problem.getStartState()
    node = (present_state, [])
    visited.append(node)

    while len(visited) > 0:  # Checking if the stack is empty
        node = visited.pop()  # Getting the last added node
        if node[0] not in checked:
            checked.append(node[0])
            if problem.isGoalState(node[0]):
                return node[1]
            for neighbor, order, cost in problem.getSuccessors(node[0]):
                next_path = node[1] + [order]
                added_node = (neighbor, next_path)
                visited.append(added_node)

    return None


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    visited = []  # Queue for nodes for actual path
    checked = []  # Stack for neighbor nodes for cross checking

    present_state = problem.getStartState()
    node = (present_state, [])
    visited.insert(0, node)

    while len(visited) > 0:
        node = visited.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        neighbors = problem.getSuccessors(node[0])
        checked.append(node[0])
        rev = range(0, len(neighbors))
        rev.reverse()
        for each in rev:
            neighbor = neighbors[each]
            if neighbor[0] not in checked:
                next_path = node[1] + [neighbor[1]]
                added_node = (neighbor[0], next_path)
                visited.insert(0, added_node)
                checked.append(neighbor[0])

    return None


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = util.PriorityQueue()
    checked = set([])
    cost_dict = {}
    seen = []
    initial_state = problem.getStartState()
    initial_state_neighbors = problem.getSuccessors(initial_state)
    neighbor_dict = {initial_state: initial_state_neighbors}

    for neighbor, order, cost in initial_state_neighbors:
        visited.push((neighbor, [order], cost), cost)
        cost_dict[neighbor] = cost
        checked.add((neighbor, order, cost))

    seen.append(problem.getStartState())

    while not visited.isEmpty():
        neighbor, order, cost = visited.pop()
        if problem.isGoalState(neighbor):
            return order
        seen.append(neighbor)
        if neighbor not in neighbor_dict.keys():
            neighbor_dict[neighbor] = problem.getSuccessors(neighbor)
        for child_neighbor, child_order, child_cost in neighbor_dict[neighbor]:
            cost_func = cost + child_cost
            valid_child = ((child_neighbor, child_order, child_cost) not in checked) and (child_neighbor not in seen)
            if valid_child:
                cost_dict[child_neighbor] = cost_func
                visited.push((child_neighbor, order + [child_order], cost + child_cost), cost_func)
                checked.add((child_neighbor, child_order, child_cost))


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    visited = util.PriorityQueue()
    checked = set([])
    cost_dict = {}
    seen = []
    initial_state = problem.getStartState()
    initial_state_neighbors = problem.getSuccessors(initial_state)
    neighbor_dict = {initial_state: initial_state_neighbors}

    for neighbor, order, cost in initial_state_neighbors:
        cost_func = cost + heuristic(neighbor, problem)
        visited.push((neighbor, [order], cost), cost_func)
        cost_dict[neighbor] = cost_func
        checked.add((neighbor, order, cost))

    seen.append(problem.getStartState())

    while not visited.isEmpty():
        neighbor, order, cost = visited.pop()
        if problem.isGoalState(neighbor):
            return order
        seen.append(neighbor)
        if neighbor not in neighbor_dict.keys():
            neighbor_dict[neighbor] = problem.getSuccessors(neighbor)
        for child_neighbor, child_order, child_cost in neighbor_dict[neighbor]:
            cost_func = cost + child_cost + heuristic(child_neighbor, problem)
            valid_child = ((child_neighbor, child_order, child_cost) not in checked) and (child_neighbor not in seen)
            if valid_child:
                cost_dict[child_neighbor] = cost_func
                visited.push((child_neighbor, order+[child_order], cost + child_cost), cost_func)
                checked.add((child_neighbor, child_order, child_cost))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
