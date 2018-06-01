import time
import random
import checker
import mdp
from utils import FIFOQueue
from copy import deepcopy
import search

ids = ["--", "--"]
WALL_CODE = 99
ACT_LIST = ["U", "D", "L", "R", "reset"]
GHOSTS = ["green", "red", "blue", "yellow"]
BLOCKED_DOTS = [21, 31, 41, 51, 71]


def count_dots(state):
    l = state.values()
    # print(state)
    # print(l)
    j = 0
    for i in l:
        if i == 11:
            j += 1

    return j


class PacmanController(mdp.MDP):
    """This class is a controller for a pacman agent."""


    def __init__(self, problem, steps):
        self.original_problem = deepcopy(problem)
        start_state, special_things = checker.problem_to_state(problem)
        self.steps = steps
        self.current_state_of_board, self.current_special_things = checker.problem_to_state(problem)
        self.accumulated_reward = 0
        self.eval = checker.Evaluator(0, problem, steps)

        self.act_list = ACT_LIST
        all_states, trans_dict, rewards = self.compute_states
        mdp.MDP.__init__(self,init=start_state, actlist=["U", "D", "R", "L"], terminals=[], transitions=trans_dict, states=all_states, )

        self.reward=rewards #mpd rewards dictionary

        self.U = mdp.value_iteration(self)

        self.pi = mdp.best_policy(self,self.U)

        # print(mdp.best_policy(self, self.U))
        print("end of initialization\n\n\n\n")
        return

    def no_dots(self,eval_state):
        for k,v in eval_state.items():
            if v%10-1==0:
                return False

        return True



    def eval_state_to_ab_state_plus_md(self,eval):
        # we assume that if pacman is dead then we return a relaxed version of the board
        min_md = 2**32-1
        if self.no_dots(eval.state):
            min_md = 0
        relaxed_eval = deepcopy(eval)
        # make sure pacman exists
        if "pacman" in eval.special_things and eval.special_things["pacman"] is not 'dead':
            pacman_place = eval.special_things["pacman"]
        # iterate over all possible ghosts
        for color in GHOSTS:
            # make sure ghost exists on board
            if color in eval.special_things.keys():
                cur_ghost_place = eval.special_things[color]
                if 'pacman_place' in locals():
                    min_md = min(min_md,abs(pacman_place[0] - cur_ghost_place[0]) + abs(pacman_place[1] - cur_ghost_place[1]))
                # convert to relaxed state without any ghosts

                relaxed_eval.state[eval.special_things[color]] = 10 + (relaxed_eval.state[eval.special_things[color]] - checker.COLOR_CODES[color])

        # relax poison dots:
        if "poison" in eval.special_things.keys():
            for cell in eval.special_things["poison"]:
                if eval.state[(cell[0], cell[1])] == 71:
                    relaxed_eval.state[(cell[0], cell[1])] = 11
                else:
                    relaxed_eval.state[(cell[0], cell[1])] = 10



        return (checker.Evaluator.state_to_agent(relaxed_eval), min_md)

    @property
    def compute_states(self):
        j = 0
        # BFS
        a_eval = deepcopy(self.eval)
        frontier = FIFOQueue()
        frontier.append(a_eval)  # Eval
        explored = set()  # ABSTRACT ( Eval.state ) + MD
        possible_state = set() # we return this so we don't have any keyerror.
        T = {}
        R = {}
        start_time = time.time()
        while frontier and j < 10000 and time.time()-start_time <40 :
            #j is just a counter to make sure we avoid infinite loop.
            j += 1
            temp_eval = frontier.pop()
            explored.add(self.eval_state_to_ab_state_plus_md(temp_eval)) # add state.

            # explore possible children of the board if they exist.

            if "pacman" in temp_eval.special_things and temp_eval.special_things["pacman"] is not 'dead':
                before_action_reward = temp_eval.accumulated_reward
                parent_state_md = self.eval_state_to_ab_state_plus_md(temp_eval)

                # children
                for action in ["U","L","R","D"]:
                    #print(action)
                    #parent

                    #child. Copy, move, get state, get reward.
                    child_eval = deepcopy(temp_eval)
                    checker.Evaluator.change_state_after_action(child_eval, action)
                    next_state_md = self.eval_state_to_ab_state_plus_md(child_eval)
                    after_action_reward = child_eval.accumulated_reward

                    # Did the action finish the board?
                    if after_action_reward - before_action_reward  >= 30:
                        #print('SOMETHING STRANGE')
                        empty_state = deepcopy(temp_eval)
                        if action == "R":
                            #print("RIGHT")
                            a = 1
                            b = 0
                        elif action == "L":
                            #print("LEFT")

                            a = -1
                            b = 0
                        elif action == "U":
                            print("UP")

                            a = 0
                            b = 1
                        elif action == "D":
                            #print("DOWN")
                            a = 0
                            b = -1
                        else:
                            # ERROR
                            #print("There is an error")
                            a=0
                            b=0
                        # update pacman's location
                        old_pacman_location = temp_eval.special_things["pacman"]
                        new_pacman_location = (temp_eval.special_things["pacman"][0]+a,temp_eval.special_things["pacman"][1]+b)
                        empty_state.special_things["pacman"] = new_pacman_location
                        empty_state.state[old_pacman_location]=10
                        empty_state.state[new_pacman_location]=66
                        R[self.eval_state_to_ab_state_plus_md(empty_state)]= after_action_reward

                        T[(parent_state_md,action)]=(1,self.eval_state_to_ab_state_plus_md(empty_state))

                        explored.add(self.eval_state_to_ab_state_plus_md(empty_state))
                        possible_state.add(self.eval_state_to_ab_state_plus_md(empty_state))
                    elif child_eval.special_things["pacman"] is not 'dead':

                        R[next_state_md] = self.set_reward(temp_eval.accumulated_reward,parent_state_md[1])

                        T[(parent_state_md, action)] = (1, next_state_md)
                        possible_state.add(next_state_md)
                        #explored.add(next_state_md)
                    # we dont want to add states where pacman is dead:
                    if "pacman" in child_eval.special_things and child_eval.special_things["pacman"] is not 'dead':
                        if next_state_md not in explored and child_eval not in frontier:
                            frontier.append(child_eval)

            print("Finished loop: States: "+ str(len(explored)) +" Queue: " + str(len(frontier)))
        print("SIZE OF EXPLORED = %s SIZE OF T = %s SIZE OF R = %s" % (len(explored),len(T),len(R)))
        return possible_state, T, R

    def set_reward(self,accumulated_reward, manhattan_distance_to_ghost):
        if manhattan_distance_to_ghost < 2:
            #
            return accumulated_reward - manhattan_distance_to_ghost
        elif manhattan_distance_to_ghost < 4:
            #neutral
            return accumulated_reward + 0.5
        elif manhattan_distance_to_ghost >= 4:
            return accumulated_reward + 1

    def T(self, state, action):
        if (state,action) in self.transitions:
            return [self.transitions[(state, action)]]
        else:
            print("Not found T")
            return [(1,state)]
    def R(self, state):
        if state in self.reward:
            return self.reward[state]
        else:
            print("Not found R")
            return 0

    def choose_next_action(self, state):
        state_of_board, special_things = checker.problem_to_state(state)
        eval_state = checker.Evaluator(0, state, 1)
        if not "pacman" in special_things:
            # check if PACMAN is still in the game
            print("HELLO buddy")
            return "reset"
        # if pacman is still in the game, then, choose best next step.
        s = self.eval_state_to_ab_state_plus_md(eval_state)
        if s  in self.pi:
            return self.pi[s]
        else:
            a = ["U","D","L","R"]
            index = random.randint(0,3)
            return a[index]

        #return (self.pi[self.eval_state_to_ab_state_plus_md(eval_state)])


