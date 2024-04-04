from collections import defaultdict

from pprint import pprint
import numpy as np
from random import sample
from scipy.special import softmax


def softmax_(env, beta, Q):
    """
    Chooses an action using softmax distribution over the available actions
    :param env: environment
    :param beta: scaling parameter of the softmax policy
    :param Q: current Q-values
    :return:
        - the chosen action
    """
    # Q values of the current state
    Q_current = Q[env.get_state()]

    # Hint: start by filtering out the non-available actions in the current state
    actions = env.available()
    encoded_actions = [env.encode_action(a) for a in actions]
    Q_available = np.array(Q_current)[encoded_actions]

    # Hint: remember to rescale all Q-values for the current state by beta
    Q_beta = [beta*val for val in Q_available]
    
    # Hint: to do a softmax operation on a set of Q-values you can use the scipy.special.softmax() function
    probas = softmax(Q_beta)
    idx = np.random.choice(a=range(len(Q_beta)), p=probas)

    return actions[idx]


def epsilon_greedy(env, epsilon, Q):
    """
    Chooses an epsilon-greedy action starting from a given state (which you can access via env.get_state()) and given a set of
    Q-values
    :param env: environment
    :param epsilon: current exploration parameter
    :param Q: current Q-values.
    :return:
        - the chosen action
    """
    # Hint: start by filtering out the non-available actions in the current state
    actions = env.available()
    encoded_actions = [env.encode_action(a) for a in actions]
    
    # Q values of the current state
    Q_current = Q[env.get_state()]
    
    # mask non available actions
    mask = np.full(env.get_num_actions(), True)
    mask[encoded_actions] = False

    if np.random.uniform(0, 1) < epsilon:
        return sample(actions, 1)[0]
    else:
        # with probability 1-epsilon choose the action with the highest immediate reward (exploitation):
        # Hint: remember to break ties randomly
        idx = np.argmax(np.ma.array(Q_current, mask=mask))
        return env.inverse_encoding(idx)


def sarsa(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", epsilon_exploration=0.1,

          epsilon_exploration_rule=None, trace_decay=0, initial_q=0):
    """
    Trains an agent using the Sarsa algorithm by playing num_episodes games until the reward states are reached
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param trace_decay: trace decay factor for eligibility traces
        If 0, sarsa(0) is implemented without any eligibility trace. If a non-zero float is given in input
        the latter represents the trace decay factor and sarsa(lambda) is implemented
    :param epsilon_exploration: parameter of the exploration policy: exploration rate or softmax_ scaling factor
        If action_policy is "epsilon_greedy":
            If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
            is taken with probability (1-epsilon_exploration)
        If action_policy is "softmax_":
            epsilon_exploration is actually beta, the scaling factor for the softmax.
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the action
        with the highest Q-value is taken with probability (1-epsilon_exploration_rule(n))
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training, with keys 'episode_rewards' and 'episode_lengths',
            and the corresponding values being a list (of length num_episodes) containing, for each episode, the reward 
            collected and the length of the episode respectively.
    """
    assert action_policy in ["epsilon_greedy", "softmax_"], "Invalid action policy is passed."
    
    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    Q = defaultdict(lambda: initial_q * np.ones(env.get_num_actions()))  # All Q-values are initialized to initial_q

    # Stats of training
    episode_rewards = np.empty(num_episodes)
    episode_lengths = np.empty(num_episodes)

    for n_ep in range(num_episodes):
        env.reset()

        # initialize the action according to the policy
        state = env.get_state()
        if action_policy=='epsilon_greedy':
            action = epsilon_greedy(env=env, epsilon=epsilon_exploration, Q=Q)
        else:
            action = softmax_(env=env, beta=epsilon_exploration, Q=Q)

        # re-initialize the eligibility traces
        traces = defaultdict(lambda: np.zeros(env.get_num_actions()))
        
        # save episode length and reward
        episode_length = 0
        episode_reward = 0

        while not env.end:
            # rescale all traces
            for s in traces.keys():
                for a, _ in enumerate(traces[s]):
                    if traces[s][a] != 0:
                       traces[s][a] = trace_decay * traces[s][a]

            # do action and get current state and reward
            state_, reward = env.do_action(action)
            encoded_action = env.encode_action(action)
            
            # choose action according to the desired policy
            if action_policy=='epsilon_greedy':
                if epsilon_exploration_rule is None:
                    action_ = epsilon_greedy(env=env, epsilon=epsilon_exploration, Q=Q)
                else:
                    action_ = epsilon_greedy(env=env, epsilon=epsilon_exploration_rule[n_ep], Q=Q)
            else:
                action_ = softmax_(env=env, beta=epsilon_exploration, Q=Q)
            encoded_action_ = env.encode_action(action_)

            # update trace of current state action pair
            traces[state][encoded_action] += 1

            # compute the target
            # Hint: all Q-values for fictitious state-action pairs are set to zero by convention
            delta = reward + gamma*Q[state_][encoded_action_] - Q[state][encoded_action]

            # update all Q-values
            for s in traces.keys():
                for a, _ in enumerate(traces[s]):
                    if traces[s][a] != 0:
                        Q[s][a] = Q[s][a] + alpha * delta * traces[s][a]

            # prepare for the next move
            episode_length += 1
            episode_reward += reward
            state = state_
            action = action_

        # save reward of the current episode
        episode_rewards[n_ep] = episode_reward
        # save length of the current episode
        episode_lengths[n_ep] = episode_length

    # save stats
    stats = {'episode_rewards':episode_rewards, 'episode_lengths':episode_lengths}
    return Q, stats


def q_learning(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", epsilon_exploration=0.1,
               epsilon_exploration_rule=None, trace_decay=0, initial_q=0):
    """
    Trains an agent using the Q-Learning algorithm by playing num_episodes games until the reward states are reached
    :param env: environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param trace_decay: trace decay factor for eligibility traces
        If 0, q_learning(0) is implemented without any eligibility trace. If a non-zero float is given in input
        the latter represents the trace decay factor and q_learning(lambda) is implemented
    :param epsilon_exploration: parameter of the exploration policy: exploration rate or softmax_ scaling factor
        If action_policy is "epsilon_greedy":
            If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
            is taken with probability (1-epsilon_exploration)
        If action_policy is "softmax_":
            epsilon_exploration is actually beta, the scaling factor for the softmax.
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the parameter for the
        exploration policy is epsilon_exploration_rule(n).
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training, with keys 'episode_rewards' and 'episode_lengths',
            and the corresponding values being a list (of length num_episodes) containing, for each episode, the reward
            collected and the length of the episode respectively.
    """

    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    Q = defaultdict(lambda: initial_q * np.ones(env.get_num_actions()))  # All Q-values are initialized to initial_q

    # Stats of training
    episode_rewards = np.empty(num_episodes)  # reward obtained for each episode
    episode_lengths = np.empty(num_episodes)  # length for each training episode

    for itr in range(num_episodes):
        env.reset()

        # initialize the action with greedy
        state = env.get_state()
        action = epsilon_greedy(env=env, epsilon=0, Q=Q)

        # re-initialize the eligibility traces
        traces = defaultdict(lambda: np.zeros(env.get_num_actions()))
        
        # save episode length and reward
        episode_length = 0
        episode_reward = 0

        while not env.end:
            # rescale all traces
            for s in traces.keys():
                for a, _ in enumerate(traces[s]):
                    if traces[s][a] != 0:
                       traces[s][a] = trace_decay * traces[s][a]

            # do action and get current state and reward
            state_, reward = env.do_action(action)
            encoded_action = env.encode_action(action)
            
            # choose action according to the desired policy
            if action_policy=='epsilon_greedy':
                if epsilon_exploration_rule is None:
                    action_ = epsilon_greedy(env=env, epsilon=epsilon_exploration, Q=Q)
                else:
                    action_ = epsilon_greedy(env=env, epsilon=epsilon_exploration_rule[itr], Q=Q)
            else:
                action_ = softmax_(env=env, beta=epsilon_exploration, Q=Q)

            # update trace of current state action pair
            traces[state][encoded_action] += 1

            # compute the target -- OFF POLICY
            # Hint: all Q-values for fictitious state-action pairs are set to zero by convention
            encoded_action_ = env.encode_action(epsilon_greedy(env=env, epsilon=0, Q=Q))
            delta = reward + gamma*Q[state_][encoded_action_] - Q[state][encoded_action]

            # update all Q-values
            for s in traces.keys():
                for a, _ in enumerate(traces[s]):
                    if traces[s][a] != 0:
                        Q[s][a] = Q[s][a] + alpha * delta * traces[s][a]

            # prepare for the next move
            episode_length += 1
            episode_reward += reward
            state = state_
            action = action_

        # save reward of the current episode
        episode_rewards[itr] = episode_reward
        # save length of the current episode
        episode_lengths[itr] = episode_length

    # save stats
    stats = {'episode_rewards':episode_rewards, 'episode_lengths':episode_lengths}
    return Q, stats


def n_step_sarsa(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", n=1,
                 epsilon_exploration=0.5, epsilon_exploration_rule=None, initial_q=0):
    """
    Trains an agent using the Sarsa algorithm by playing num_episodes games until the reward states are reached
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param n: for n = 1 standard Sarsa(0) is recovered, otherwise n-step Sarsa is implemented
    :param epsilon_exploration: parameter of the exploration policy: exploration rate or softmax_ scaling factor
        If action_policy is "epsilon_greedy":
            If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
            is taken with probability (1-epsilon_exploration)
        If action_policy is "softmax_":
            epsilon_exploration is actually beta, the scaling factor for the softmax.
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training, with keys 'episode_rewards' and 'episode_lengths',
            and the corresponding values being a list (of length num_episodes) containing, for each episode, the reward
            collected and the length of the episode respectively.
    """
    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    Q = defaultdict(lambda: initial_q * np.ones(env.get_num_actions()))  # All Q-values are initialized to initial_q

    # Stats of training
    episode_rewards = np.empty(num_episodes)
    episode_lengths = np.empty(num_episodes)

    # Hint: it may be useful to compute the weight of the rewards, something like
    reward_weights = np.array([gamma ** i for i in range(n)])

    for itr in range(num_episodes):

        env.reset()

        # initialize a queue for state action pairs and rewards of the current episode
        state = env.get_state()
        if action_policy=='epsilon_greedy':
            action = epsilon_greedy(env=env, epsilon=epsilon_exploration, Q=Q)
        else:
            action = softmax_(env=env, beta=epsilon_exploration, Q=Q)

        # save episode length and reward
        episode_length = 0
        episode_reward = 0

        # counter
        t = 0
        encoded_actions = []
        states = []
        rewards = []
        while not env.end:
            # update the counter
            t += 1

            # Move according to the policy
            # Save obtained reward
            state_, reward = env.do_action(action)
            encoded_action = env.encode_action(action)

            # track the current reward, state and action
            if t > n:
                encoded_actions.pop(0)
                states.pop(0)
                rewards.pop(0)
            encoded_actions += [encoded_action]
            states += [state]
            rewards += [reward]

            # choose action according to the desired policy
            if action_policy=='epsilon_greedy':
                if epsilon_exploration_rule is None:
                    action_ = epsilon_greedy(env=env, epsilon=epsilon_exploration, Q=Q)
                else:
                    action_ = epsilon_greedy(env=env, epsilon=epsilon_exploration_rule[itr], Q=Q)
            else:
                action_ = softmax_(env=env, beta=epsilon_exploration, Q=Q)
            encoded_action_ = env.encode_action(action_)

            # Updating all the Q values within the n step from the reward
            # Not only a single Q Value n step away from the reward
            if t >= n:
                for i in range(n):
                    # Hint: all Q-values for fictitious state-action pairs are set to zero by convention
                    delta = np.dot(reward_weights, rewards) + gamma**n*Q[state_][encoded_action_] - Q[states[i]][encoded_actions[i]]

                    # update Q-value of state-action pair which is n steps away from the reward
                    Q[states[i]][encoded_actions[i]] = Q[states[i]][encoded_actions[i]] + alpha * delta            
            
            # prepare next move
            episode_length += 1
            episode_reward += reward
            state = state_
            action = action_

        # save reward of the current episode
        episode_rewards[itr] = episode_reward
        # save length of the current episode
        episode_lengths[itr] = episode_length

    # save stats
    stats = {'episode_rewards':episode_rewards, 'episode_lengths':episode_lengths}
    return Q, stats
