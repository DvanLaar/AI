# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        # This is our value iteration code
        # it iterates 'iterations' times, and each iteration
        # it updates the values based on the possible outcomes.
        for i in range(iterations): 
            valuesPrime = self.values.copy()
            for state in self.mdp.getStates():  
                if self.mdp.isTerminal(state):  
                    valuesPrime[state] = 0
                    continue
                action = self.computeActionFromValues(state)
                valuesPrime[state] = self.computeQValueFromValues(state, action)
            self.values = valuesPrime
            

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        
        som = 0 # start at 0
        for Pstate, prob in self.mdp.getTransitionStatesAndProbs(state, action): # for every term
            reward = self.mdp.getReward(state, action, Pstate) # find the value
            value = self.values[Pstate]                         # of the term
            som += prob * (reward + value * self.discount)      # and add it to the sum
        return som
            

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        bestAction = None
        for action in self.mdp.getPossibleActions(state): #loop for every action
            if bestAction == None:  
                bestAction = action                 #set bestaction to this
                continue
            if self.computeQValueFromValues(state, action) > self.computeQValueFromValues(state, bestAction): #if this action is better then the best action
                bestAction = action                                                                            # set bestaction to this
        return bestAction           #done

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
