import time
import random
import mdp
from copy import deepcopy

ids = ["--", "--"]
PILL_CONSTANT = 1
PILL_CODE = 11
WALL_CODE = 99
EMPTY_CODE = 10
PACMAN_CODE = 66
RESET_CONSTANT = -20
END_OF_LEVEL_CONSTANT = 50
PROBABILITIES = {"red": 0.9, "green": 0.7, "blue": 0.4, "yellow": 0.4}
BLOCKING_CODES = (20, 21, 30, 31, 40, 41, 50, 51, 99)
LOSS_INDEXES =   (20, 21, 30, 31, 40, 41, 50, 51, 71, 77)
COLOR_CODES = {"red": 50, "green": 40, "blue": 20, "yellow": 30}
DEBUG_PRINT = True




def problem_to_state(problem):
    """

    Args:
        problem: pacman problem as tuple of tuples, as described in PDF

    Returns: internal state representation, as dictionary of {coordinate: code} and
    dictionary of objects of special interest (pacman, ghosts, poison) as {name: coordinate}

    """
    state_to_return = {}
    special_things = {"poison": []}

    for number_of_row, row in enumerate(problem):
        for number_of_column, cell in enumerate(row):
            state_to_return[(number_of_row, number_of_column)] = cell
            if cell == 66:
                special_things["pacman"] = (number_of_row, number_of_column)
                print("Pacman position is:", number_of_row, number_of_column)
            elif cell == 50 or cell == 51:
                special_things["red"] = (number_of_row, number_of_column)
            elif cell == 40 or cell == 41:
                special_things["green"] = (number_of_row, number_of_column)
            elif cell == 30 or cell == 31:
                special_things["yellow"] = (number_of_row, number_of_column)
            elif cell == 20 or cell == 21:
                special_things["blue"] = (number_of_row, number_of_column)
            elif cell == 77 or cell == 71:
                special_things["poison"].append((number_of_row, number_of_column))
    return state_to_return, special_things

def move_pacman(state, special_things, action):
        next_tile = None
        current_tile_x, current_tile_y = special_things["pacman"]
        if action == "U":
            next_tile = (current_tile_x - 1, current_tile_y)
        if action == "R":
            next_tile = (current_tile_x, current_tile_y + 1)
        if action == "D":
            next_tile = (current_tile_x + 1, current_tile_y)
        if action == "L":
            next_tile = (current_tile_x, current_tile_y - 1)

        assert next_tile is not None

        # wall
        if state[next_tile] == 99:
            return special_things["pacman"]

        # walkable tile
        if state[next_tile] == 10 or state[next_tile] == 11:
            state[next_tile] = 66
            state[(current_tile_x, current_tile_y)] = 10
            special_things["pacman"] = next_tile
            return special_things["pacman"]

        # ghosts and poison
        if state[next_tile] in LOSS_INDEXES:
            state[next_tile] = 88
            state[(current_tile_x, current_tile_y)] = 10
            #special_things["pacman"] = "dead"
            special_things["pacman"] = next_tile
            return special_things["pacman"]



class PacmanController(mdp.MDP):
    """This class is a controller for a pacman agent."""

    def __init__(self, state, steps):
        """Initialize controller for given the initial setting.
        This method MUST terminate within the specified timeout."""
        reward_grid ={}
        # we initialize a R grid as follows:
        # pill = +1, wall = 0, ghost or poison = -1, and empty cell = 0
        for number_of_row, row in enumerate(state):
            #row_list = []
            for number_of_column, cell in enumerate(row):
                if cell == 66:
                    pacman_location = [number_of_row,number_of_column]
                if cell == PILL_CODE:
                    reward_grid[(number_of_row,number_of_column)] = 1
                    #row_list.append(1)
                elif cell == WALL_CODE:
                    reward_grid[(number_of_row,number_of_column)] = None
                    #row_list.append(None)
                elif cell in LOSS_INDEXES:
                    reward_grid[(number_of_row,number_of_column)] = -1
                    #row_list.append(-1)
                else:
                    reward_grid[(number_of_row,number_of_column)] = 0
                    #row_list.append(0)
            #reward_grid.append(row_list)
        self.state, self.special_things = problem_to_state(state)
        states = []
        actlist = ["U", "D", "L", "R"]
        for cell in self.state:
            if (self.state[cell] != WALL_CODE) and not(self.state[cell] in self.special_things):
                states.append((cell))
        super().__init__(terminals=None, init=pacman_location, gamma=0.1, actlist=actlist, states=states)
        self.steps = steps
        self.reward = reward_grid
        self.U = mdp.value_iteration(self)
        self.policy_eval = mdp.policy_iteration(self)
        self.best_policy = mdp.best_policy(self,self.U)


        print('COMPLETE init ')



    def T(self, state, action):
        s1 = deepcopy((self))
        # updated the future state based on the pacman position passed in state
        s1.special_things["pacman"] = state
        s1.state[(self.special_things["pacman"])] = EMPTY_CODE
        s1.state[(s1.special_things["pacman"])] = PACMAN_CODE
        if action is None:
            return [(0.0, state)]
        else:
            return [(0.25, move_pacman(s1.state, s1.special_things, action))]

    def actions(self, state):
        """Set of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
#        if state in self.terminals:
 #           return [None]
  #      else:
        return self.actlist

    def go(self, state, action):

        new_location = []
        if action == "U":
            new_location[0] = self.special_things['pacman'][0] + 1
            new_location[1] = self.special_things['pacman'][1] + 0
        elif action == "D":
            new_location[0] = self.special_things['pacman'][0] - 1
            new_location[1] = self.special_things['pacman'][1] + 0
        elif action == "R":
            new_location[0] = self.special_things['pacman'][0] + 0
            new_location[1] = self.special_things['pacman'][1] + 1
        elif action == "L":
            new_location[0] = self.special_things['pacman'][0] + 0
            new_location[1] = self.special_things['pacman'][1] - 1
        else:
            print("in <go>: illegal action detected")
            return None



    def choose_next_action(self, state):
        """Choose next action for pacman controller given the current state.
        Action should be returned in the format described previous parts of the project.
        This method MUST terminate within the specified timeout."""
        self.state, self.special_things = problem_to_state(state)
        # check if PACMAN is still in the game
        if not "pacman" in self.special_things:
            return "reset"
        #return self.policy_eval[self.special_things["pacman"]]
        return self.best_policy[self.special_things["pacman"]]
        print('COMPLETE choose_next_action')


