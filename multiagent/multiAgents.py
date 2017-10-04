# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        total_score = 0
        ghost_distance_list = []

        for each in newGhostStates:
            ghost_distance_list.append(manhattanDistance(each.getPosition(), newPos))
        food_list = newFood.asList()

        food_distance_list = []

        for each in food_list:
            food_distance_list.append(manhattanDistance(newPos, each))
        if len(food_distance_list) > 0 and len(ghost_distance_list) > 0:
            food_score = 2.25 / (min(food_distance_list) ** 2)
        else:
            food_score = 0
        if min(ghost_distance_list) > float(3):
            ghost_score = 3
        elif min(ghost_distance_list) < 1.1:
            ghost_score = -100
        else:
            ghost_score = min(ghost_distance_list)
        total_score += food_score + ghost_score + successorGameState.getScore()
        return total_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        self.agent_count = gameState.getNumAgents()

        if self.depth == 0:
            return float('inf')

        neg_infinity = float('-inf')
        possible_actions = gameState.getLegalActions(0)
        if len(possible_actions) == 0:
            return self.evaluationFunction()

        for each in possible_actions:
            next_man = gameState.generateSuccessor(0, each)
            temp_min = self.get_minimum_food(next_man, self.depth, 1)
            if temp_min >= neg_infinity:
                best_move = each
                neg_infinity = temp_min

        return best_move

    def get_minimum_food(self, state, depth, agent_index):
        if depth == 0:
            return self.evaluationFunction(state)

        final_min = float('inf')
        possible_actions = state.getLegalActions(agent_index)
        if len(possible_actions) == 0:
            return self.evaluationFunction(state)

        for each in possible_actions:
            next_man = state.generateSuccessor(agent_index, each)
            if agent_index + 1 == self.agent_count:
                temp_food = self.get_maximum_food(next_man, depth-1)
            else:
                temp_food = self.get_minimum_food(next_man, depth, agent_index+1)
            final_min = min(final_min, temp_food)
        return final_min

    def get_maximum_food(self, state, depth):
        if depth == 0:
            return self.evaluationFunction(state)

        final_max = float('-inf')
        possible_actions = state.getLegalActions(0)
        if len(possible_actions) == 0:
            return self.evaluationFunction(state)

        for each in possible_actions:
            next_man = state.generateSuccessor(0, each)
            temp_food = self.get_minimum_food(next_man, depth, 1)
            final_max = max(final_max, temp_food)

        return final_max



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.agent_count = gameState.getNumAgents()

        alpha = float('-inf')
        beta = float('inf')

        if self.depth == 0:
            return float('inf')

        best_move = Directions.STOP
        neg_infinity = float('-inf')
        possible_actions = gameState.getLegalActions(0)
        if len(possible_actions) == 0:
            return self.evaluationFunction()

        for each in possible_actions:
            next_man = gameState.generateSuccessor(0, each)
            temp_min = self.get_minimum_food(next_man, self.depth, alpha, beta, 1)
            if temp_min >= neg_infinity:
                best_move = each
                neg_infinity = temp_min
            if neg_infinity > beta:
                return neg_infinity
            alpha = max(alpha, neg_infinity)
        return best_move

    def get_minimum_food(self, state, depth, alpha, beta, agent_index):
        if depth == 0:
            return self.evaluationFunction(state)

        final_min = float('inf')
        possible_actions = state.getLegalActions(agent_index)
        if len(possible_actions) == 0:
            return self.evaluationFunction(state)

        for each in possible_actions:
            next_man = state.generateSuccessor(agent_index, each)
            if agent_index + 1 == self.agent_count:
                temp_food = self.get_maximum_food(next_man, depth-1, alpha, beta)
            else:
                temp_food = self.get_minimum_food(next_man, depth, alpha, beta, agent_index+1)
            final_min = min(final_min, temp_food)
            if alpha > final_min:
                return final_min
            beta = min(beta, final_min)

        return final_min

    def get_maximum_food(self, state, depth, alpha, beta):
        if depth == 0:
            return self.evaluationFunction(state)

        final_max = float('-inf')
        possible_actions = state.getLegalActions(0)
        if len(possible_actions) == 0:
            return self.evaluationFunction(state)

        for each in possible_actions:
            next_man = state.generateSuccessor(0, each)
            temp_food = self.get_minimum_food(next_man, depth, alpha, beta, 1)
            final_max = max(final_max, temp_food)
            if beta < final_max:
                return final_max
            alpha = max(final_max, alpha)

        return final_max

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.agent_count = gameState.getNumAgents()

        if self.depth == 0:
            return float('inf')

        neg_infinity = float('-inf')
        possible_actions = gameState.getLegalActions(0)
        best_move = Directions.STOP
        if len(possible_actions) == 0:
            return self.evaluationFunction()

        for each in possible_actions:
            next_man = gameState.generateSuccessor(0, each)
            temp_min = self.get_minimum_food(next_man, self.depth, 1)
            if temp_min > neg_infinity:
                best_move = each
                neg_infinity = temp_min

        return best_move

    def get_minimum_food(self, state, depth, agent_index):
        if depth == 0:
            return self.evaluationFunction(state)

        possible_actions = state.getLegalActions(agent_index)
        if len(possible_actions) == 0:
            return self.evaluationFunction(state)

        final_sum = 0
        final_count = 0

        for each in possible_actions:
            next_man = state.generateSuccessor(agent_index, each)
            if agent_index + 1 == self.agent_count:
                temp_food = self.get_maximum_food(next_man, depth-1)
            else:
                temp_food = self.get_minimum_food(next_man, depth, agent_index+1)
            final_sum += temp_food
            final_count += 1

        return final_sum/final_count

    def get_maximum_food(self, state, depth):
        if depth == 0:
            return self.evaluationFunction(state)

        final_max = float('-inf')
        possible_actions = state.getLegalActions(0)
        if len(possible_actions) == 0:
            return self.evaluationFunction(state)

        for each in possible_actions:
            next_man = state.generateSuccessor(0, each)
            temp_food = self.get_minimum_food(next_man, depth, 1)
            final_max = max(final_max, temp_food)

        return final_max


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

