import matplotlib.pyplot as plt
from typing import Tuple

from environments.abstract_environment import AbstractEnvironment

State = Tuple[float, float]
Action = int
Reward = int

MOVES = [
    (0, 1),
    ((1/2)**0.5, (1/2)**0.5),
    (1, 0),
    ((1/2)**0.5, -(1/2)**0.5),
    (0, -1),
    (-(1/2)**0.5, -(1/2)**0.5),
    (-1, 0),
    (-(1/2)**0.5, (1/2)**0.5),
]

# noinspection PyAttributeOutsideInit
class UMaze(AbstractEnvironment[State, Action, Reward]):
    def __init__(self, step_size=0.2):
        self._step_size = step_size
        self.reset()

    @property
    def end(self) -> bool:
        return self._end

    def do_action(self, action: Action) -> Tuple[State, Reward]:
        if self._end:
            raise ValueError('One of the two goal states has already been reached, please reset the game!')
        #if not type(action) is Action:
        #    raise TypeError(f"Should be of the action type {Action}, is {type(action)}")

        move = MOVES[action]
        new_state = (self._state[0] + self._step_size*move[0], self._state[1] + self._step_size*move[1])
        self._history.append(self._state)

        if UMaze.is_outside_of_area(*new_state):
            # reject moves that go out of the allowed area
            new_state = self._state
            reward = 0
        elif UMaze.is_in_objective(*new_state):
            reward = 1
            self._end = True
        else:
            reward = 0

        self._state = new_state
        self._last_reward = reward

        return new_state, reward

    @staticmethod
    def is_in_objective(x, y):
        """
        :returns: true if the position is in the target (circle of radius 1/2 around the origin)
        """
        return x*x + y*y <= 1/4

    @staticmethod
    def is_outside_of_area(x, y):
        """
        :returns: true if the position is outside the accessible area
        """
        # Check outer border
        if abs(x) > 5 or abs(y) > 5:
            return True
        # U sides
        if 1.5 <= abs(x) <= 2.5 and -2.5 < y < 2:
            return True
        # U bottom
        if abs(x) <= 1.5 and -2.5 <= y <= -1.5:
            return True
        return False

    def get_state(self) -> State:
        return self._state

    def reward(self) -> Reward:
        return self._last_reward

    def reset(self):
        self._end = False
        self._state = (0.0, 0.0)
        self._history = []

    def render(self):
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

        # Plot objective
        ax.add_patch(plt.Circle((0, 0), 0.5, color="green"))
        # Plot U shape
        ax.add_patch(plt.Rectangle((-2.5, -2.5), 1, 4.5, color="red"))
        ax.add_patch(plt.Rectangle((1.5, -2.5), 1, 4.5, color="red"))
        ax.add_patch(plt.Rectangle((-1.5, -2.5), 3, 1, color="red"))

        # Plot points
        for i in range(-6, 7): # TODO: these points need to be pulled from the ModelParams!
            for j in range(-6, 7):
                plt.plot(i, j, 'b.', markersize=2)

        # Plot outside boundaries
        plt.plot([-5, -5, 5, 5, -5], [-5, 5, 5, -5, -5], 'r')

        # Plot history
        for i in range(len(self._history)):
            pos1 = self._history[i]
            pos2 = self._history[i+1] if i+1 < len(self._history) else self._state
            plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'b')

        # Plot current position
        plt.plot(*self._state, 'kx', markersize=10)

        plt.show()
