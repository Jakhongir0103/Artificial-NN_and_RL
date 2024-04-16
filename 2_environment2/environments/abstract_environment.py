from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple

State = TypeVar("State")
Reward = TypeVar("Reward")
Action = TypeVar("Action")


class AbstractEnvironment(ABC, Generic[State, Action, Reward]):
    """
    Abstract environment class, can be instantiated in the following way:
    ```
    class ExamplEnvironment(AbstractEnviroment[state_type, action_type, reward_type])
    ```
    where `state_type`, `action_type` can be anything. For `reward_type`, prefer numerical types like `float` and `int`.
    Also, see examples/example_inheritance.py
    """

    @property
    @abstractmethod
    def end(self) -> bool:
        """
        Boolean attribute of the class that signals whether we are in a terminal state.
        Can simply be implemented as an attribute, does not have to be a property.
        :return: bool, True if the environment is in a terminal state, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def do_action(self, action: Action) -> Tuple[State, Reward]:
        """
        Executes the action provided in its argument on the environment, and it should return a tuple of the state
        (deepcopied/immutable) and the reward
        :return: Tuple[State, Action]
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> State:
        """
        Returns a deepcopy of the current state, or an immutable type
        :return: State
        """
        raise NotImplementedError

    @abstractmethod
    def reward(self) -> Reward:
        """
        Returns the last received reward
        :return: Reward
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Resets the environment
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        """
        Renders the current state, either textually or visually with Matplotlib
        :return: None
        """
        raise NotImplementedError