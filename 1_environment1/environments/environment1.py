from .abstract_environment import AbstractEnvironment
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import matplotlib.cm as cmx

State = Tuple[int, int]
Action = str
Reward = int


class TMaze(AbstractEnvironment[State, Action, Reward]):
    """
    This class implements a T Maze with a certain number of states before arriving in the state denoted by A in the
    below scheme. Such a number of states is stored in the attribute "self.vertical_depth" and must be passed as an
    input when constructing the environment. This means that "self.vertical_depth" represents the number of needed steps
    for an agent to arrive in A from the initial state of each episode. Similarly, the attribute "self.horizontal_depth"
    represents the number of states from A to one of the two rewarded states at the endpoints of the T shape.

    EXAMPLE: in Week 1 exercise session, you will investigate some standard discrete settings reinforcement learning in
             the one-step horizon setting. This means reward is received after the very first action is taken. For this,
             you should set self.vertical_depth = 0 and self.horizontal_depth = 1, so every episode starts in A and the
             whole learning process depends only on whether the agent chooses the action "Move Left" or "Move Right".

    The encoding of the states of the environment follows the following convention: the self._start state always
    corresponds to the origin (0, 0). From each state one can take one of the following actions: ('u', 'd', 'l',
    'r') but a ValueError is raised if, for example, action 'down' is taken from the origin. The state A is encoded
    as the state (0, self._vertical_depth) and the two goal states are respectively (+- self._horizontal_depth,
    self._vertical_depth).

    EXAMPLE: for the one-step horizon, A is the state (0, 0) and the two goal states are (-1, 0) and (1, 0).

            -------------------------
    r = 1  |           A            |   r = 2
            ---------       ---------
                    |       |
                    |       |
                    |       |
                    |       |
                    |       |
                    ---------
                      start
    """

    def __init__(self, horizontal_depth, vertical_depth):
        self._end = False
        self._current_state = (0, 0)
        self._vertical_depth = vertical_depth  # steps needed to arrive in A from self.start
        self._horizontal_depth = horizontal_depth
        self._start = (0, 0)  # default starting state is the origin
        self._goal_states = ((-horizontal_depth, vertical_depth), (horizontal_depth, vertical_depth))
        self.num_states = self._vertical_depth + 2 * self._horizontal_depth + 1
        self._direct_path_len = self._vertical_depth + self._horizontal_depth
        self.num_actions = 4

    @property
    def end(self) -> bool:
        """
        :return: self._end
        """
        return self._end

    def get_state(self) -> State:
        """
        Get current state of the environment
        :return:
        """
        return self._current_state

    def highest_reward(self):
        return 2

    def reset(self):
        """
        Reset game
        :return:
        """
        self._end = False
        self._current_state = self._start

    def get_initial_state(self) -> State:
        """
        Get starting point of the episode
        :return:
        """
        return self._start

    def get_num_actions(self) -> int:
        """
        Get possible number of actions in any state
        :return:
        """
        return self.num_actions

    def get_num_states(self) -> int:
        """
        Returns number of states of the environment
        """
        return self.num_states

    def direct_path_len(self) -> int:
        """
        Returns length of direct path to the rewarded states from the origin
        """
        return self._direct_path_len

    def available(self):
        """
        Returns the available actions in the current state
        :return: List containing the available actions
        """
        if self._current_state == (0, self._vertical_depth):  # current state is A
            if self._vertical_depth != 0:
                return ['d', 'l', 'r']
            else:
                return ['l', 'r']
        elif self._current_state[0] != 0 and self._current_state[1] == self._vertical_depth:  # on the horizontal branch
            return ['l', 'r']
        elif self._current_state[0] == 0 and self._current_state[1] > 0 and \
                self._current_state[1] != self._vertical_depth:  # on the vertical branch, but not the origin
            return ['u', 'd']
        elif self._current_state == self._start:  # origin
            return ['u']

    def do_action(self, action: Action) -> Tuple[Tuple[int, int], int]:
        """
        Performs an action from the current state of the environment
        :param action: chosen action
        :return: state of the environment and reward after the taken action
        """
        if self._end:
            raise ValueError('One of the two goal states has already been reached, please reset the game!')
        if not type(action) is Action:
            raise TypeError(f"Should be of the action type {Action}")
        if action == 'd' and self._current_state == self._start:
            raise ValueError("Cannot go back from the starting node")
        if action == 'u' and self._current_state[1] == self._vertical_depth:
            raise ValueError("Cannot go up from the current state")
        if (action == 'r' or action == 'l') and self._current_state[0] == 0 \
                and self._current_state[1] != self._vertical_depth:
            raise ValueError("Cannot go left or right before arriving in state A")

        # The following instructions are executed if the chosen action is available
        if action == 'l':
            next_state = (self._current_state[0] - 1, self._current_state[1])
        elif action == 'r':
            next_state = (self._current_state[0] + 1, self._current_state[1])
        elif action == 'u':
            next_state = (self._current_state[0], self._current_state[1] + 1)
        elif action == 'd':
            next_state = (self._current_state[0], self._current_state[1] - 1)

        self._current_state = next_state
        return self._current_state, self.reward()

    def reward(self) -> Reward:
        """
        :return: reward, see reward scheme in the class description
        """
        if self._current_state == (-self._horizontal_depth, self._vertical_depth):
            self._end = True
            return 1
        elif self._current_state == (self._horizontal_depth, self._vertical_depth):
            self._end = True
            return 2
        else:
            return 0

    @staticmethod
    def encode_action(action):
        """
        Map from action string to integers for default dict compatibility
        :param action: string valued action
        :return:
        """
        if action == 'u':
            return 0
        if action == 'd':
            return 1
        if action == 'l':
            return 2
        if action == 'r':
            return 3

    @staticmethod
    def inverse_encoding(action) -> Action:
        """
        Inverse map of self.encode_action(action)
        :param action: integer valued action
        :return:
        """
        if action == 0:
            return 'u'
        if action == 1:
            return 'd'
        if action == 2:
            return 'l'
        if action == 3:
            return 'r'

    def neighbours(self):
        """
        Returns the neighbours of the current state. The order is the one corresponding to the available actions
        returned by the self.available method.
        :return: List containing the neighbouring states
        """
        if self._current_state[0] == 0 and self._current_state[1] == self._vertical_depth:
            if self._vertical_depth == 0:
                return [(self._current_state[0] - 1, self._vertical_depth),
                        (self._current_state[0] + 1, self._vertical_depth)]
            else:
                return [(self._current_state[0], self._current_state[1] - 1),
                        (self._current_state[0] - 1, self._vertical_depth),
                        (self._current_state[0] + 1, self._vertical_depth)]
        elif self._current_state[0] != 0 and self._current_state[1] == self._vertical_depth:
            return [(self._current_state[0] - 1, self._vertical_depth),
                    (self._current_state[0] + 1, self._vertical_depth)]
        elif self._current_state[0] == 0 and self._current_state[1] != self._vertical_depth:
            if self._current_state[1] == 0:
                return [(self._current_state[0], self._current_state[1] + 1)]
            else:
                return [(self._current_state[0], self._current_state[1] + 1),
                        (self._current_state[0], self._current_state[1] - 1)]

    def render(self, Q=None, show=True):
        """
        Render, shows a figure of the current state of the environment
        :return:
        """
        if Q is None:
            fig, ax = plt.subplots(figsize=(self.get_num_states()/2, self.get_num_states()/2))
        else:
            fig, ax = plt.subplots(figsize=(1.3 * self.get_num_states()/2, self.get_num_states()/2))
        plt.vlines(-0.25, -0.25, self._vertical_depth - 0.25)
        plt.vlines(0.25, -0.25, self._vertical_depth - 0.25)
        plt.vlines(-self._horizontal_depth - 0.25, self._vertical_depth-0.25, self._vertical_depth+0.25)
        plt.vlines(self._horizontal_depth + 0.25, self._vertical_depth-0.25, self._vertical_depth+0.25)
        plt.hlines(self._vertical_depth - 0.25, -self._horizontal_depth - 0.25, -0.25)
        plt.hlines(self._vertical_depth - 0.25, 0.25, self._horizontal_depth + 0.25)
        plt.hlines(self._vertical_depth + 0.25, -self._horizontal_depth - 0.25, self._horizontal_depth + 0.25)
        plt.hlines(-0.25, -0.25, 0.25)

        if Q is not None:
            # Create a continuous norm to map from data points to colors
            norm = colors.Normalize(vmin=0.5, vmax=2)
            scalarMap = cmx.ScalarMappable(norm=norm, cmap="viridis")

        if Q is None:
            for ver in range(self._vertical_depth):
                plt.plot([0, 0], [ver, ver + 1], color="black", marker="o", linewidth=2, markersize=5)

            for hor in range(self._horizontal_depth):
                plt.plot([hor, hor+1], [self._vertical_depth, self._vertical_depth], color="black", marker="o", linewidth=3, markersize=5)
                plt.plot([-hor, -hor-1], [self._vertical_depth, self._vertical_depth], color="black", marker="o", linewidth=3, markersize=5)
        else:
            for ver in range(self._vertical_depth):
                q = Q[(0, ver)][self.encode_action("u")]
                colorVal = scalarMap.to_rgba(q)
                plt.arrow(x=0, y=ver, dx=0, dy=1/2, width=0.1, head_width=0.2, color=colorVal, length_includes_head=True)
                if ver > 0:
                    q = Q[(0, ver)][self.encode_action("d")]
                    colorVal = scalarMap.to_rgba(q)
                    plt.arrow(x=0, y=ver, dx=0, dy=-1/2, width=0.1, head_width=0.2,
                              color=colorVal, length_includes_head=True)

            for hor in range(-self._horizontal_depth + 1, self._horizontal_depth):
                q = Q[(hor, self._vertical_depth)][self.encode_action("r")]
                colorVal = scalarMap.to_rgba(q)
                plt.arrow(x=hor, y=self._vertical_depth, dx=1/2, dy=0, head_width=0.2, width=0.1,
                          color=colorVal, length_includes_head=True)
                q = Q[(hor, self._vertical_depth)][self.encode_action("l")]
                colorVal = scalarMap.to_rgba(q)
                plt.arrow(x=hor, y=self._vertical_depth, dx=-1/2, dy=0, head_width=0.2, width=0.1,
                          color=colorVal, length_includes_head=True)

            q = Q[(0, self._vertical_depth)][self.encode_action("d")]
            colorVal = scalarMap.to_rgba(q)
            plt.arrow(x=0, y=self._vertical_depth, dx=0, dy=-1/2, width=0.1, head_width=0.2,
                      color=colorVal, length_includes_head=True)

        # Plotting the current position of the agent, together with the starting and goal state
        if Q is None:
            (x, y) = self._start
            plt.plot(x, y, color='green', marker='o', markersize=10)
            (x, y) = self._current_state
            plt.plot(x, y, color='red', marker="X", markersize=10)
            (x, y) = self._goal_states[0]
            plt.plot(x, y, color='blue', marker='*', markersize=10)
            (x, y) = self._goal_states[1]
            plt.plot(x, y, color='blue', marker='*', markersize=10)
            # Figure specifications
            plt.axis('off')
            if show:
                plt.show()
            else:
                plt.close(fig)
        else:
            (x, y) = self._start
            plt.plot(x, y, color='green', marker='o', markersize=7)
            (x, y) = self._goal_states[0]
            plt.plot(x, y, color='blue', marker='*', markersize=10)
            (x, y) = self._goal_states[1]
            plt.plot(x, y, color='blue', marker='*', markersize=10)
            cbar = plt.colorbar(scalarMap)
            cbar.set_label('Q-values', rotation=90, fontsize=15)
            plt.axis('off')
            if show:
                plt.show()
            else:
                plt.close(fig)

        return fig, ax
