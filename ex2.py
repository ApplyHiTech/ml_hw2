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


class PacmanController(mdp.MDP):
    """This class is a controller for a pacman agent."""

    def __init__(self, problem, steps):
        self.original_problem = deepcopy(problem)
        start_state, special_things = checker.problem_to_state(problem)
        self.steps = steps
        self.current_state_of_board, self.current_special_things = checker.problem_to_state(problem)
        self.eval = checker.Evaluator(0, problem, steps)
        self.act_list = ACT_LIST
        all_states, trans_dict, rewards = self.compute_states
        print(all_states)
        print(rewards)
        mdp.MDP.__init__(self,init=start_state, actlist=["U", "D", "R", "L"], terminals=[], transitions=trans_dict, states=all_states,gamma=0.01 )

        self.reward=rewards #mpd rewards dictionary

        self.U = mdp.value_iteration(self)

        self.pi = mdp.best_policy(self,self.U)

        # print(mdp.best_policy(self, self.U))
        print("end of initialization\n\n\n\n")
        return


    def eval_state_to_ab_state_plus_md(self,eval): #
        # we assume that if pacman is dead then we return a relaxed version of the board
        min_md = 2**32-1

        relaxed_eval = deepcopy(eval)
        # make sure pacman exists
        if "pacman" in eval.special_things and eval.special_things["pacman"] is not 'dead':
            pacman_place = eval.special_things["pacman"]
        # iterate over all possible ghosts
        num_ghosts = 0
        for color in GHOSTS:
            # make sure ghost exists on board
            if color in eval.special_things.keys():
                num_ghosts+=1
                cur_ghost_place = eval.special_things[color]
                if 'pacman_place' in locals():
                    min_md = min(min_md,abs(pacman_place[0] - cur_ghost_place[0]) + abs(pacman_place[1] - cur_ghost_place[1]))
                # convert to relaxed state without any ghosts

                relaxed_eval.state[eval.special_things[color]] = 10 + (relaxed_eval.state[eval.special_things[color]] - checker.COLOR_CODES[color])


        # relax poison dots:
        num_pills = 0
        if "poison" in eval.special_things.keys():
            for cell in eval.special_things["poison"]:
                num_pills+=1

                if eval.state[(cell[0], cell[1])] == 71:
                    relaxed_eval.state[(cell[0], cell[1])] = 11
                else:
                    relaxed_eval.state[(cell[0], cell[1])] = 10




        return (checker.Evaluator.state_to_agent(relaxed_eval), min_md, num_pills, num_ghosts)

    @property
    def compute_states(self):
        # BFS
        a_eval = deepcopy(self.eval)
        frontier = FIFOQueue()
        min_front_reward = 0
        frontier.append(a_eval)  # Eval
        explored = set()  # ABSTRACT ( Eval.state ) + MD
        possible_state = set() # we return this so we don't have any keyerror.

        possible_state.add(("DEADEND"))
        for i in possible_state:
            print(i)
        T = {}
        R = {}
        start_time = time.time()
        flag = True
        while flag and frontier and time.time() - start_time < 40 :
            temp_eval = frontier.pop()
            explored.add(self.eval_state_to_ab_state_plus_md(temp_eval)) # add state.

            # explore possible children of the board if they exist.

            if "pacman" in temp_eval.special_things and temp_eval.special_things["pacman"] is not 'dead':
                before_action_h_value = self.set_h(temp_eval)
                #print(temp_eval.special_things)

                #print(before_action_h_value)

                parent_state_md = self.eval_state_to_ab_state_plus_md(temp_eval)
                parent_state_reward = temp_eval.accumulated_reward
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

                    if after_action_reward - parent_state_reward  >= 30:
                        flag = False # found a solution
                        print('SOMETHING STRANGE')
                        empty_state = deepcopy(temp_eval)
                        if action == "R":
                            #print("RIGHT")
                            a = 0
                            b = 1
                        elif action == "L":
                            #print("LEFT")

                            a = 0
                            b = -1
                        elif action == "U":
                            print("UP")

                            a = -1
                            b = 0
                        elif action == "D":
                            #print("DOWN")
                            a = 1
                            b = 0
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
                        R[self.eval_state_to_ab_state_plus_md(empty_state)]= after_action_reward*100
                        print("REWARD %s " %after_action_reward)

                        T[(parent_state_md,action)]=(1,self.eval_state_to_ab_state_plus_md(empty_state))

                        explored.add(self.eval_state_to_ab_state_plus_md(empty_state))
                        print(self.eval_state_to_ab_state_plus_md(empty_state))
                        print("ADDED")
                        print(len(possible_state))
                        possible_state.add(self.eval_state_to_ab_state_plus_md(empty_state))
                        print(len(possible_state))

                    elif child_eval.special_things["pacman"] is not 'dead':

                        R[next_state_md] = self.set_h(temp_eval)

                        T[(parent_state_md, action)] = (1, next_state_md)
                        possible_state.add(next_state_md)
                        #explored.add(next_state_md)
                    # we dont want to add states where pacman is dead:
                    if "pacman" in child_eval.special_things and child_eval.special_things["pacman"] is not 'dead':
                        if next_state_md not in explored and child_eval not in frontier:
                            if child_eval.accumulated_reward >= min_front_reward-1:
                                min_front_reward = max(child_eval.accumulated_reward,min_front_reward)
                                frontier.append(child_eval)
                                print("Min Reward %s" % min_front_reward)


            print("Finished loop: States: "+ str(len(explored)) +" Queue: " + str(len(frontier)))
        print("SIZE OF EXPLORED = %s SIZE OF T = %s SIZE OF R = %s" % (len(explored),len(T),len(R)))
        return possible_state, T, R

    def numb_of_ghosts(self):
        i =0
        for k in self.current_special_things:
            if k in checker.LOSS_INDEXES:
                i+=1

        return i

    def set_h(self,temp_eval): # heuristic
        accum_reward = temp_eval.accumulated_reward

        state, min_dist, num_pills, num_ghosts = self.eval_state_to_ab_state_plus_md(temp_eval)
        #print("acc reward %s" % accum_reward)
        #print("num ghosts %s"   % num_ghosts)
        #print("num pills %s " % num_pills)
        #print("min_dist %s " % min_dist)
        return 10 * accum_reward
        #return 10*accum_reward +1*min_dist + 50*num_pills + 5*num_ghosts

    def T(self, state, action):
        if (state,action) in self.transitions:
            return [self.transitions[(state, action)]]
        else:
            #print("Not found T")
            return [(1,("DEADEND"))]
    def R(self, state):
        if state in self.reward:
            return self.reward[state]
        else:
            print("Not found R")
            print(state)
            self.reward[state]=0
            return 0

    def find_min_md_from_ghosts(self, eval_state):
        min_md = 2**32-1
        if not "pacman" in eval_state.special_things:
            return -100
        pacman_place = eval_state.special_things["pacman"]
        for color in GHOSTS:
            # make sure ghost exists on board
            if color in eval_state.special_things.keys():
                cur_ghost_place = eval_state.special_things[color]
                min_md = min(min_md,abs(pacman_place[0] - cur_ghost_place[0]) + abs(pacman_place[1] - cur_ghost_place[1]))

        return min_md

    def choose_next_action(self, state):
        state_of_board, special_things = checker.problem_to_state(state)
        eval_state = checker.Evaluator(0, state, 1)
        if not "pacman" in special_things:
            # check if PACMAN is still in the game
            return "reset"
        # if pacman is still in the game, then, choose best next step.
        s = self.eval_state_to_ab_state_plus_md(eval_state)
        if s  in self.pi:
            new_min_md = 0
            # check if we need to update R based on Ghost location:
            min_md = self.find_min_md_from_ghosts(eval_state)
            # we check if there any ghosts on the board, and if they are very close.
            if min_md != -100 and min_md <=2:
                print("performing update to R")
                # start scanning for a better position
                for action in ["U","L","R","D"]:
                    child_eval = deepcopy(eval_state)
                    checker.Evaluator.change_state_after_action(child_eval, action)
                    temp_new_md = self.find_min_md_from_ghosts(child_eval)
                    if temp_new_md != -100 and temp_new_md > new_min_md:
                        new_min_md = temp_new_md
                        next_state_md = self.eval_state_to_ab_state_plus_md(child_eval)
                        self.rewards[next_state_md] = self.rewards[next_state_md] + 10*new_min_md
                # TODO: we might be yeilding a state that didnt exist before
                self.U = mdp.value_iteration(self)
                self.pi = mdp.best_policy(self,self.U)
            return self.pi[s]
        else:
            a = ["U","D","L","R"]
            print("random chosen")
            # maybe here we should go into a simple dfs to find rest of the route to finish the board? @meir
            index = random.randint(0,3)
            return a[index]

        #return (self.pi[self.eval_state_to_ab_state_plus_md(eval_state)])
