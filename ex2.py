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
PILLS = [77,71]
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

        # STATE LOOKS LIKE:
        # accum_reward +1*+ 50*num_pills + 5*num_ghosts
        # min_dist to nearest ghost
        # far_dist to ghost
        # num_of_ghosts
        # num_pills
        # dist_to_nearest_pill
        # dist_to_farthest_pill
        # dist_left_wall
        # dist_right_wall
        # dist_up_wall
        # dist_down_wall

        mdp.MDP.__init__(self,init=start_state, actlist=["U", "D", "R", "L"], terminals=[], transitions=trans_dict, states=all_states, )

        self.reward=rewards #mpd rewards dictionary

        self.U = mdp.value_iteration(self)

        self.pi = mdp.best_policy(self,self.U)

        # print(mdp.best_policy(self, self.U))
        print("end of initialization\n\n\n\n")
        return


    def eval_state_to_ab_state_plus_md(self,a_eval): #
        # we assume that if pacman is dead then we return a relaxed version of the board
        dist_nearest_ghost= 2**10-1
        dist_farthest_ghost = 0
        num_ghosts = 0
        num_pills = 0
        dist_nearest_pill = 0
        dist_farthest_pill = 0
        #dist_left_wall = 0
        #dist_right_wall = 0
        #dist_up_wall = 0
        #dist_down_wall = 0
        #print("A EVAL ACCUM: %s" % a_eval.accumulated_reward)
        accum_reward = a_eval.accumulated_reward

        relaxed_eval = deepcopy(a_eval)

        # make sure pacman exists
        if "pacman" in a_eval.special_things and a_eval.special_things["pacman"] is not 'dead':
            pacman_place = a_eval.special_things["pacman"]

        # Compute Ghosts count, Ghost min and max dist to pacman; relax board
        # Do the same for Poison pills.
        for special_obj in a_eval.special_things:
            if special_obj in GHOSTS:
                num_ghosts+=1
                ghost_loc = a_eval.special_things[special_obj]
                dist = manhattan_distance(pacman_place,ghost_loc)
                if dist < dist_nearest_ghost:
                    dist_nearest_ghost =dist
                elif dist > dist_farthest_ghost:
                    dist_farthest_ghost= dist
                # Updated relaxed board so that it removes ghosts.
                relaxed_eval.state[a_eval.special_things[special_obj]] = 10 + (
                            relaxed_eval.state[a_eval.special_things[special_obj]] - checker.COLOR_CODES[special_obj])

            elif special_obj in PILLS:
                num_pills+=1
                pill_loc = a_eval.special_things[special_obj]
                dist = manhattan_distance(pacman_place, pill_loc)
                if dist < dist_nearest_pill:
                    dist_nearest_ghost = dist
                elif dist > dist_farthest_pill:
                    dist_farthest_ghost = dist
                # Updated relaxed board so that it removes poison pills.
                if a_eval.state[pill_loc] == 71:
                    relaxed_eval.state[pill_loc] = 11
                else:
                    relaxed_eval.state[pill_loc] = 10

        return (checker.Evaluator.state_to_agent(relaxed_eval),num_ghosts,dist_nearest_ghost,dist_farthest_ghost,num_pills,dist_nearest_pill,dist_farthest_pill,accum_reward)

    @property
    def compute_states(self):
        # BFS
        a_eval = deepcopy(self.eval)
        frontier = FIFOQueue()
        min_front_reward = 0
        frontier.append(a_eval)  # Eval
        explored = set()  # ABSTRACT ( Eval.state ) + MD
        possible_state = set() # we return this so we don't have any keyerror.
        T = {}
        R = {}
        start_time = time.time()
        while frontier and time.time() - start_time < 10 :
            temp_eval = frontier.pop()
            explored.add(self.eval_state_to_ab_state_plus_md(temp_eval)) # add state.
            print("START LOOP accumulated reward = %s" % temp_eval.accumulated_reward)

            # explore possible children of the board if they exist.

            if "pacman" in temp_eval.special_things and temp_eval.special_things["pacman"] is not 'dead':
                before_action_h_value = self.set_h(temp_eval)
                parent_state_md = self.eval_state_to_ab_state_plus_md(temp_eval)

                # children of current state.
                for action in ["U","L","R","D"]:
                    #print(action)
                    #parent

                    #child. Copy, move, get state, get reward.
                    child_eval = deepcopy(temp_eval)
                    # 2 - Move
                    checker.Evaluator.change_state_after_action(child_eval, action)
                    # 3 - Get state
                    next_state_md = self.eval_state_to_ab_state_plus_md(child_eval)
                    # 4 - Get reward
                    after_action_reward = self.set_h(child_eval)

                    # Did the action finish the board?
                    if checker.Evaluator.finished_the_game(child_eval):
                        print("Print HUH?")
                    #if after_action_reward - before_action_h_value  >= 30:
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

                        R[next_state_md] = after_action_reward

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



    def set_h(self,temp_eval): # heuristic
        state, num_ghosts, dist_nearest_ghost, dist_farthest_ghost, num_pills, dist_nearest_pill, dist_farthest_pill, accum_reward = self.eval_state_to_ab_state_plus_md(temp_eval)
        print("SET H acc reward %s" % accum_reward)
        #print("num ghosts %s"   % num_ghosts)
        #print("num pills %s " % num_pills)
        #print("min_dist %s " % dist_nearest_ghost)
        # accum_reward +1*+ 50*num_pills + 5*num_ghosts
        # min_dist to nearest ghost
        # far_dist to ghost
        # num_of_ghosts
        # num_pills
        # dist_to_nearest_pill
        # dist_to_farthest_pill
        # dist_left_wall
        # dist_right_wall
        # dist_up_wall
        # dist_down_wall
        # features = [reward, num_ghosts, min_dist_ghost, max_dist_ghost, num_pills, min_dist_pill, max_dist_pill]
        weights =    [15,2,2,0.1,50,5,2]
        f = [accum_reward,num_ghosts, dist_nearest_ghost, dist_farthest_ghost, num_pills, dist_nearest_pill, dist_farthest_pill]
        x = sum(i[0] * i[1] for i in zip(weights, f))
        print("h value: %s " % x)
        return x
        #return 10*accum_reward +1*dist_nearest_ghost+1*_
        #return 10*accum_reward +1*min_dist + 50*num_pills + 5*num_ghosts

    def T(self, state, action):
        if (state,action) in self.transitions:
            return [self.transitions[(state, action)]]
        else:
            #print("Not found T")
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

def manhattan_distance(place_0, place_1):
    x=place_0[0]
    y=place_0[1]
    x2=place_1[0]
    y2=place_1[1]
    return abs(x-x2) + abs(y-y2)


