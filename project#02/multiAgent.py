# multiAgents.py
# ------------------------------
# Pavlos Spanoudakis (sdi1800184)
# ------------------------------
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
from pacman import GameState



class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        Chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some
        X in the set {NORTH, SOUTH, WEST, EAST, STOP}.
        """
        # Collect legal moves
        legalMoves = gameState.getLegalActions()

        # Evaluate each legal move using the evaluation function
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        A better evaluation function that prioritizes food proximity,
        avoids ghosts, and seeks scared ghosts.
        """
        # Generate the successor state for the action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # Initialize score based on the successor state score
        if successorGameState.isWin():
            return float('inf')  # Winning state is the highest possible score
        if successorGameState.isLose():
            return float('-inf')  # Losing state is the lowest possible score
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        # Reward for being closer to food
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        if foodDistances:
            score += 15.0 / (min(foodDistances) + 2)  # Inverse to prioritize closer food

        # Penalize being close to active ghosts
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)

            if scaredTime > 0:
                # Reward getting closer to scared ghosts
                score += 60.0 / (ghostDist + 1)
            else:
                # Penalize proximity to active ghosts more
                if ghostDist < 2:
                    score -= 1000 / (ghostDist + 1)  # Strong penalty for being very close to an active ghost
                elif ghostDist < 4:
                    score -= 200 / (ghostDist + 1)  # Moderate penalty for being near an active ghost

        # Avoid stopping if there are better options
        if action == Directions.STOP:
            score -= 10  # Penalize stopping unless necessary
        
        return score


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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def minimax(agentIndex, depth, state):
            # Terminal state: game won, lost, or depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman's turn (Maximizing player, agentIndex = 0)
            if agentIndex == 0:
                return self.maxValue(agentIndex, depth, state)

            # Ghosts' turn (Minimizing players, agentIndex > 0)
            else:
                return self.minValue(agentIndex, depth, state)

        def maxValue(agentIndex, depth, state):
            v = float('-inf')
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:  # No legal actions
                return self.evaluationFunction(state)

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, minimax(1, depth, successor))  # Next is the first ghost (agentIndex = 1)
            return v

        def minValue(agentIndex, depth, state):
            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:  # No legal actions
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():  # Last ghost; Pacman's turn next
                nextAgent = 0
                depth += 1

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgent, depth, successor))
            return v

        # Find the best action for Pacman (agentIndex = 0)
        legalMoves = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)  # Start with the first ghost (agentIndex = 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your AlphaBeta agent (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the AlphaBeta action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def alphaBeta(agentIndex, depth, state, alpha, beta):
            # Terminal state: game won, lost, or depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman's turn (Maximizing player, agentIndex = 0)
            if agentIndex == 0:
                return self.maxValue(agentIndex, depth, state, alpha, beta)

            # Ghosts' turn (Minimizing players, agentIndex > 0)
            else:
                return self.minValue(agentIndex, depth, state, alpha, beta)

        def maxValue(agentIndex, depth, state, alpha, beta):
            v = float('-inf')
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:  # No legal actions
                return self.evaluationFunction(state)

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, alphaBeta(1, depth, successor, alpha, beta))  # Next is the first ghost (agentIndex = 1)
                if v > beta:  # Beta cut-off
                    return v
                alpha = max(alpha, v)  # Update alpha
            return v

        def minValue(agentIndex, depth, state, alpha, beta):
            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:  # No legal actions
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():  # Last ghost; Pacman's turn next
                nextAgent = 0
                depth += 1

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, alphaBeta(nextAgent, depth, successor, alpha, beta))
                if v < alpha:  # Alpha cut-off
                    return v
                beta = min(beta, v)  # Update beta
            return v

        # Find the best action for Pacman (agentIndex = 0)
        legalMoves = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = alphaBeta(1, 0, successor, alpha, beta)  # Start with the first ghost (agentIndex = 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)  # Update alpha

        return bestAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your Expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the Expectimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def expectimax(agentIndex, depth, state):
            # Terminal state: game won, lost, or depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman's turn (Maximizing player, agentIndex = 0)
            if agentIndex == 0:
                return self.maxValue(agentIndex, depth, state)

            # Ghosts' turn (Chance nodes, agentIndex > 0)
            else:
                return self.expectationValue(agentIndex, depth, state)

        def maxValue(agentIndex, depth, state):
            v = float('-inf')
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:  # No legal actions
                return self.evaluationFunction(state)

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, expectimax(1, depth, successor))  # Next is the first ghost (agentIndex = 1)
            return v

        def expectationValue(agentIndex, depth, state):
            v = 0
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:  # No legal actions
                return self.evaluationFunction(state)

            # Here we calculate the expected value for the ghost's random moves
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                # Assuming equal probability for each action by ghosts
                v += expectimax(agentIndex + 1, depth, successor)  # Recursive call for next agent (ghost or pacman)
            
            return v / len(legalActions)  # Average value for the chance node (expectation)

        # Find the best action for Pacman (agentIndex = 0)
        legalMoves = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(1, 0, successor)  # Start with the first ghost (agentIndex = 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    This evaluation function balances:
    - Avoiding non-scared ghosts at all costs.
    - Prioritizing food, capsules, and scared ghosts efficiently.
    - Keeping a strong emphasis on game progress and winning.
    """

    # Check if the game is in a terminal state
    if currentGameState.isWin():
        return float('inf')  # Winning state is the highest possible score
    if currentGameState.isLose():
        return float('-inf')  # Losing state is the lowest possible score

    # Extract key information
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    # Initialize evaluation score with current game score
    evalScore = currentGameState.getScore()

    # Weights for different components
    foodWeight = 10.0
    capsuleWeight = 50.0
    scaredGhostWeight = 200.0
    ghostAvoidanceWeight = -500.0

    # Food: Encourage Pacman to eat food
    if food:
        foodDistances = [util.manhattanDistance(pacmanPos, foodPos) for foodPos in food]
        evalScore += foodWeight / (1 + min(foodDistances))  # Prioritize closer food

    # Capsules: Encourage Pacman to eat capsules
    if capsules:
        capsuleDistances = [util.manhattanDistance(pacmanPos, capsulePos) for capsulePos in capsules]
        evalScore += capsuleWeight / (1 + min(capsuleDistances))  # Prioritize closer capsules

    # Ghosts: Avoid active ghosts, chase scared ones
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        scaredTime = ghostState.scaredTimer
        distance = util.manhattanDistance(pacmanPos, ghostPos)

        if scaredTime > 0:  # Scared ghost: chase it
            evalScore += scaredGhostWeight / (1 + distance)
        elif distance < 2:  # Active ghost: avoid at all costs
            evalScore += ghostAvoidanceWeight / (1 + distance)

    # Encourage Pacman to clear the board faster by prioritizing progress
    evalScore -= len(food) * 10  # Penalty for remaining food
    evalScore -= len(capsules) * 20  # Penalty for remaining capsules

    return evalScore

# Abbreviation
better = betterEvaluationFunction
