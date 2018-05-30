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
        all_states, trans_dict, rewards = self.compute_states()
        mdp.MDP.__init__(self,init=start_state, actlist=["U", "D", "R", "L"], terminals=None, transitions=trans_dict, states=all_states, )

        self.reward=rewards #mpd rewards dictionary

        self.U = mdp.value_iteration(self)

        self.pi = mdp.best_policy(self,self.U)

        # print(mdp.best_policy(self, self.U))
        print("end of initialization\n\n\n\n")
        return

    def eval_state_to_ab_state_plus_md(self,eval):
        min_md = 2**32-1
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

    def compute_states(self):
        j = 0
        # BFS
        a_eval = deepcopy(self.eval)
        frontier = FIFOQueue()
        frontier.append(a_eval)  # Eval
        explored = set()  # ABSTRACT ( Eval.state ) + MD
        T = {}
        R = {}
        while frontier and j < 10000:
            #j is just a counter to make sure we avoid infinite loop.
            j += 1
            temp = frontier.pop()

            if "pacman" in temp.special_things and temp.special_things["pacman"] is not 'dead':
                explored.add(self.eval_state_to_ab_state_plus_md(temp))
                #explored.add(checker.Evaluator.state_to_agent(temp))

                # children
                curr_evalU = deepcopy(temp)
                curr_evalL = deepcopy(temp)
                curr_evalR = deepcopy(temp)
                curr_evalD = deepcopy(temp)
                # apply different actions to each child
                tmp_reward = curr_evalU.accumulated_reward
                curr_state = self.eval_state_to_ab_state_plus_md(temp)

                checker.Evaluator.change_state_after_action(curr_evalU, "U")
                if curr_evalU.accumulated_reward - tmp_reward >= 30:
                    # we force apply the state after applying the specific action which caused a reset.
                    empty_state = deepcopy(temp)
                    empty_state.special_things["pacman"] = (temp.special_things["pacman"][0]-1, temp.special_things["pacman"][1])
                    empty_state.state[temp.special_things["pacman"]] = 10
                    empty_state.state[empty_state.special_things["pacman"]] = 66
                    R[checker.Evaluator.state_to_agent(empty_state)] = curr_evalU.accumulated_reward
                    T[(curr_state, "U")] = (1, self.eval_state_to_ab_state_plus_md(empty_state))

                else:
                    T[(curr_state, "U")] = (1, self.eval_state_to_ab_state_plus_md(curr_evalU))

                tmp_reward = curr_evalL.accumulated_reward
                checker.Evaluator.change_state_after_action(curr_evalL, "L")
                if curr_evalL.accumulated_reward - tmp_reward >= 30:
                    # we force apply the state after applying the specific action which caused a reset.
                    empty_state = deepcopy(temp)
                    empty_state.special_things["pacman"] = (temp.special_things["pacman"][0], temp.special_things["pacman"][1]-1)
                    empty_state.state[temp.special_things["pacman"]] = 10
                    empty_state.state[empty_state.special_things["pacman"]] = 66
                    R[checker.Evaluator.state_to_agent(empty_state)] = curr_evalL.accumulated_reward
                    T[(curr_state, "L")] = (1, self.eval_state_to_ab_state_plus_md(empty_state))

                else:
                    T[(curr_state, "L")] = (1, self.eval_state_to_ab_state_plus_md(curr_evalL))


                tmp_reward = curr_evalR.accumulated_reward
                checker.Evaluator.change_state_after_action(curr_evalR, "R")
                # handle special case where all the dots are eaten but the method resets the board:
                if curr_evalR.accumulated_reward - tmp_reward >= 30:
                    # we force apply the state after applying the specific action which caused a reset.
                    empty_state = deepcopy(temp)
                    empty_state.special_things["pacman"] = (temp.special_things["pacman"][0], temp.special_things["pacman"][1] + 1)
                    empty_state.state[temp.special_things["pacman"]] = 10
                    empty_state.state[empty_state.special_things["pacman"]] = 66
                    R[checker.Evaluator.state_to_agent(empty_state)] = curr_evalR.accumulated_reward
                    T[(curr_state, "R")] = (1, self.eval_state_to_ab_state_plus_md(empty_state))

                else:
                    T[(curr_state, "R")] = (1, self.eval_state_to_ab_state_plus_md(curr_evalR))



                tmp_reward = curr_evalD.accumulated_reward
                checker.Evaluator.change_state_after_action(curr_evalD, "D")
                if curr_evalD.accumulated_reward - tmp_reward >= 30:
                    # we force apply the state after applying the specific action which caused a reset.
                    empty_state = deepcopy(temp)
                    empty_state.special_things["pacman"] = (temp.special_things["pacman"][0]+1, temp.special_things["pacman"][1])
                    empty_state.state[temp.special_things["pacman"]] = 10
                    empty_state.state[empty_state.special_things["pacman"]] = 66
                    R[checker.Evaluator.state_to_agent(empty_state)] = curr_evalD.accumulated_reward
                    T[(curr_state, "D")] = (1, self.eval_state_to_ab_state_plus_md(empty_state))

                else:
                    T[(curr_state, "D")] = (1, self.eval_state_to_ab_state_plus_md(curr_evalD))

                child_evals = [curr_evalU, curr_evalL, curr_evalR, curr_evalD]

                curr_state = self.eval_state_to_ab_state_plus_md(temp)

                R[curr_state] = temp.accumulated_reward

                #T[(curr_state, "U")] = (1, checker.Evaluator.state_to_agent(curr_evalU))
                #T[(curr_state, "L")] = (1, checker.Evaluator.state_to_agent(curr_evalL))
                #T[(curr_state, "R")] = (1, checker.Evaluator.state_to_agent(curr_evalR))
                for child in child_evals:
                    # If all dots are eaten, the board resets.
                    # So, we check to see if any action resulted in a reset board.
                    # If yes, we have found a solution.
                    if child.state == self.eval.state and j > 1:
                        # if a specific action doesnt change the board (such as moving into a wall) then the above if condition applies,
                        # but empty_state doesn't exisrt
                        if 'empty_state' in locals():
                            explored.add(self.eval_state_to_ab_state_plus_md(empty_state))
                        print("WAHOO\n\n\n\n")
                        print("FINISHED------")
                        #print(R)
                        #R[curr_state]+=50

                        #return explored, T, R We should exit here, but then we have an issue with missing some states.

                    # we dont want to add states where pacman is dead:
                    if "pacman" in child.special_things and child.special_things["pacman"] is not 'dead':
                        if self.eval_state_to_ab_state_plus_md(child) not in explored and child not in frontier:
                            frontier.append(child)

            print("Finished loop: States: "+ str(len(explored)) +" Queue: " + str(len(frontier)))
        #print(R)
        return explored, T, R

    def T(self, state, action):
        if (state,action) in self.transitions:
            return [self.transitions[(state, action)]]
        else:
            return [(1,state)]
    def R(self, state):
        if state in self.reward:
            return self.reward[state]
        else:
            return 0

    def choose_next_action(self, state):
        state_of_board, special_things = checker.problem_to_state(state)
        if not "pacman" in special_things:
            # check if PACMAN is still in the game
            print("HELLO buddy")
            return "reset"
        # if pacman is still in the game, then, choose best next step.
        return (self.pi[state])

    def actions(self,state):

        return ["U","D","L","R"]

    def value_iteration(self, epsilon=0.001):
        """Solving an MDP by value iteration. [Figure 17.4]"""

        U1 = {s: 0 for s in self.states}
        print(len(U1))
        R, T, gamma = self.R, self.T, self.gamma
        j=0
        while j<100:
            j+=1
            U = U1.deepcopy()
            delta = 0
            for s in self.states:
                print (s)
                #print (self.actions(s))
                max_val = 0
                for a in ["U", "D", "L", "R"]:
                    p,s1 = self.T[(s,a)]

                    if max_val < p*U[s1]:

                        max_val = p*U[s1]

                U1[s] = self.R[s] + gamma * max_val

                delta = max(delta, abs(U1[s] - U[s]))

            if delta < epsilon * (1 - gamma) / gamma:
                return U
