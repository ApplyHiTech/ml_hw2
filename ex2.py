import time
import random
import checker
import mdp
from copy import deepcopy

ids = ["--", "--"]
WALL_CODE = 99
ACT_LIST = ["U","D","L","R","reset"]
GHOSTS = ["green","red","blue","yellow"]
BLOCKED_DOTS = [11,21,31,41,51,71]
class PacmanController(mdp.MDP):
    """This class is a controller for a pacman agent."""

    def __init__(self, state, steps):
        self.actlist = ACT_LIST
        self.states,self.rewards,self.max_dots,self.max_ghosts = self.generate_ab_states(state)
        self.steps = steps
        return

    def generate_ab_states(self,state):
        states = []
        ab_state,special_things = self.get_abstract_state(state)
        num_dots = ab_state[0]
        num_ghosts = ab_state[1]
        max_dots = num_dots
        max_ghosts = num_ghosts
        rewards = {}

        for i in range(0,num_dots+1):
            for j in range(0,num_ghosts+1):
                # state vector = num_dots, num_ghosts,
                new_state = [(i,j)]
                if i == 0:
                    rewards[(i,j)] = 50+max_dots
                else:
                    rewards[(i,j)] = max_dots-i #number of dots left.

                states+=new_state

        return states,rewards,max_dots,max_ghosts


    # Takes a state and generates a feature vector for the abstract state.
    def get_abstract_state(self,state):
        state_of_board, special_things = checker.problem_to_state(state)
        num_ghosts = 0
        num_dots = 0
        for i in special_things:
            if i in GHOSTS:
                num_ghosts+=1

        for i in state_of_board:
            if state_of_board[i] == 11 or state_of_board[i] in BLOCKED_DOTS:
                num_dots +=1
        # dist_nearest_dot = nearest_dot(pacman,state_of_board)
        # state feature vector
        ab_state = (num_dots, num_ghosts)
        return ab_state,special_things



    def choose_next_action(self, state):
        ab_state,special_things = self.get_abstract_state(state)
        print(ab_state)

        if not "pacman" in special_things:
            # check if PACMAN is still in the game
            return "reset"
        # if pacman is still in the game, then, choose best next step.
        # this is temporarily defaulted to be Up for all moves!
        return "U"

