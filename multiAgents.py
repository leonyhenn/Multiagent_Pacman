# cd /Users/study/Test/384/a3/multiagent
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
        #print scores
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        #print "legalMoves[chosenIndex]",legalMoves[chosenIndex]
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
        curPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFood_list = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        curFood = currentGameState.getFood()
        foodNum = successorGameState.getNumFood()
        #1. avoid ghost to where ghost might step on
        ghostPositions = successorGameState.getGhostPositions()

        pacman_ghost_closest = 10000
        for gp in ghostPositions:
          if manhattanDistance(newPos,gp) < pacman_ghost_closest: 
            pacman_ghost_closest = manhattanDistance(newPos,gp)
        
        closest_food = 10000
        for food in (newFood_list):
          if newFood[food[0]][food[1]] == True:
            if manhattanDistance(newPos,food) < closest_food:
              closest_food = manhattanDistance(newPos,food)
        if curFood[newPos[0]][newPos[1]] == True:
            closest_food = 0
        if pacman_ghost_closest <=1:
          return -10000
        else:
          return -closest_food


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
        """
        "*** YOUR CODE HERE ***"

        bestMove,score = DFminimax(self,gameState,0,0)
        #print "bestMove",bestMove
        return bestMove

def DFminimax(self,gameState,agentIndex,depth):
  bestMove = None
  if gameState.isWin() or gameState.isLose() or depth>=self.depth:
    # if agentIndex == gameState.getNumAgents() -1:
    #   print self.evaluationFunction(gameState)
    #   print "depth",depth
    return bestMove,self.evaluationFunction(gameState)
  score = float("Inf")
  if agentIndex == 0:
    score = -float("Inf")

  moves = gameState.getLegalActions(agentIndex)

  for move in moves:
    
    next_gameState = gameState.generateSuccessor(agentIndex,move)
    if agentIndex == 0:#max player

      next_move, next_score = DFminimax(self,next_gameState,agentIndex+1,depth)
      # print "pacman start"
      # print depth
      # print "next_move",next_move,"next_score",next_score
      # print "pacman end"
      if next_score > score:
        score = next_score
        bestMove = move
    else:
      if agentIndex == (gameState.getNumAgents() -1):
        next_agent = 0
        depth_next = depth+1
      else:
        next_agent = agentIndex + 1
        depth_next = depth
      next_move, next_score = DFminimax(self,next_gameState,next_agent,depth_next)
      # print "ghost start",agentIndex
      # print depth
      # print "next_move",next_move,"next_score",next_score
      # print "ghost end",agentIndex
      if next_score < score:
        score = next_score
        bestMove = move
  #print bestMove, score
  return bestMove,score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        bestAction,score = ABhelper(self,gameState,0,0,-float("Inf"),float("Inf"))
        return bestAction
def ABhelper(self,gameState,agentIndex,depth,alpha,beta):
  bestMove = None
  if gameState.isWin() or gameState.isLose() or depth>=self.depth:
    return bestMove,self.evaluationFunction(gameState)
  score = float("Inf")
  if agentIndex == 0:
    score = -float("Inf")

  moves = gameState.getLegalActions(agentIndex)
  for move in moves:
    next_gameState = gameState.generateSuccessor(agentIndex,move)
    if agentIndex == 0:#max player
      next_move, next_score = ABhelper(self,next_gameState,agentIndex+1,depth,alpha,beta)
      if next_score > score:
        score = next_score
        bestMove = move
      alpha = max(score,alpha)
      if score >= beta:
        return bestMove,score
    else:
      if agentIndex == (gameState.getNumAgents() -1):
        next_agent = 0
        depth_next = depth+1
      else:
        next_agent = agentIndex + 1
        depth_next = depth
      next_move, next_score = ABhelper(self,next_gameState,next_agent,depth_next,alpha,beta)
      if next_score < score:
        score = next_score
        bestMove = move
      beta = min(score,beta)
      if score <= alpha:
        return bestMove,score
  return bestMove,score





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
        bestAction,score = Exmax(self,gameState,0,0)
        return bestAction
        
def Exmax(self,gameState,agentIndex,depth):
  bestMove = None
  if gameState.isWin() or gameState.isLose() or depth>=self.depth:
    return bestMove,self.evaluationFunction(gameState)
  score = 0
  if agentIndex == 0:
    score = -float("Inf")

  moves = gameState.getLegalActions(agentIndex)

  for move in moves:
    next_gameState = gameState.generateSuccessor(agentIndex,move)
    if agentIndex == 0:#max player
      next_move, next_score = Exmax(self,next_gameState,agentIndex+1,depth)
      if next_score > score:
        score = next_score
        bestMove = move
    else:
      if agentIndex == (gameState.getNumAgents() -1):
        next_agent = 0
        depth_next = depth+1
      else:
        next_agent = agentIndex + 1
        depth_next = depth
      next_move, next_score = Exmax(self,next_gameState,next_agent,depth_next)
      score = score + (1/float(len(moves))) * next_score
  return bestMove,score

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curFood_list = curFood.asList()
    capPos = currentGameState.getCapsules()
    curFood_list.extend(capPos)
    for cap in capPos:
      curFood[cap[0]][cap[1]] = True
    score = currentGameState.getScore()
    #1. avoid ghost to where ghost might step on
    ghostPositions = currentGameState.getGhostPositions()

    closest_capsule = 10000
    if len(capPos) == 0:
      closest_capsule = 0
    for capsule in capPos:
      if manhattanDistance(capsule,curPos) < closest_capsule:
        closest_capsule = manhattanDistance(capsule,curPos)


    pacman_ghost_closest = 10000
    for gp in ghostPositions:
      if Euclid_dist(curPos,gp) < pacman_ghost_closest: 
        pacman_ghost_closest = Euclid_dist(curPos,gp)


    man_pacman_ghost_closest = 10000
    for gp in ghostPositions:
      if manhattanDistance(curPos,gp) < man_pacman_ghost_closest: 
        man_pacman_ghost_closest = manhattanDistance(curPos,gp)

    num_food_left = 0
    closest_food = 10000
    for food in (curFood_list):
      if curFood[food[0]][food[1]] == True:
        num_food_left += 1
        if Euclid_dist(curPos,food) < closest_food:
          closest_food = Euclid_dist(curPos,food)

    man_closest_food = 10000
    for food in (curFood_list):
      if curFood[food[0]][food[1]] == True:
        num_food_left += 1
        if manhattanDistance(curPos,food) < man_closest_food:
          man_closest_food = manhattanDistance(curPos,food)

    if curFood[curPos[0]][curPos[1]] == True:
        closest_food = 0
        man_closest_food
    if man_pacman_ghost_closest <=1:
      return -1000000
    else:
      #return score - man_pacman_ghost_closest 10/12 1die 1019 avg MAYBE
      # return score - pacman_ghost_closest 4/12 NO
      # return score - closest_food NO
      # return score - num_food_left*closest_food NO
      # return score - closest_capsule MAYBE 
      # return score - len(capPos)*closest_capsule  MAYBE
      # return score - man_closest_food NO
      # return score - num_food_left*man_closest_food MAYBE
      # return score - 1/float(closest_food) NO
      # return score - 1/float(man_closest_food) NO

      # return score + man_pacman_ghost_closest 10/12 0die 845.3 avg MAYBE
      # return score + pacman_ghost_closest 10/12 0die 836.0 avg MAYBE
      # return score + closest_food NO
      # return score + num_food_left*closest_food NO
      # return score + closest_capsule  NO
      # return score + len(capPos)*closest_capsule  NO
      # return score + man_closest_food NO
      # return score + num_food_left*man_closest_food NO
      # return score + 1/float(closest_food) MAYBE 8/12 2die avg780 a lot of 1000+
      # return score + 1/float(man_closest_food)  THE ONE !!!!
      # return score + 1/float(man_closest_food) + man_pacman_ghost_closest 10/12 0die
      # return score + 1/float(man_closest_food) + pacman_ghost_closest  10/12 0die 775avg 
      #return score + 1/float(man_closest_food) + + 1/float(closest_food) THEONE!!!! 1088 avg
      return score + 1/float(man_closest_food) + + 1/float(closest_food) + 1/float(man_pacman_ghost_closest)
      # NEW THE ONE!!! 1160 avg 0 die super fast
      #return score + 1/float(man_closest_food) + + 1/float(closest_food) + 1/float(man_pacman_ghost_closest)+ 1/float(pacman_ghost_closest) 1die 1019avg
      #return score + 1/float(man_closest_food) + + 1/float(closest_food) + 1/float(pacman_ghost_closest) 1die 905avg
      #return score + 1/float(man_closest_food) + + 1/float(closest_food) + 1/float(man_pacman_ghost_closest) + 1/float(closest_capsule+1) 1127avg 0die 12/12

      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 
      # return score - man_pacman_ghost_closest - len(capPos) - closest_capsule 

# Abbreviation
better = betterEvaluationFunction

def Euclid_dist(pos1,pos2):
    import math
    return int(math.sqrt(pow((pos1[0] - pos2[0]),2)+pow((pos1[1] - pos2[1]),2)))