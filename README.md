# EX-6: Implementation-of-MC-prediction-for-estimating-the-action-value-function

## Aim:

To implement the Monte Carlo (MC) Prediction algorithm for estimating the action-value function \(Q(s,a)\) using sampled episodes and to analyze the learned action values in a Grid World environment.

---

## Objective:

- To understand Monte Carlo Prediction for action-value estimation.
- To estimate the action-value function \(Q(s,a)\).
- To learn state-action values from complete episodes.
- To evaluate the quality of actions under a given policy.

---

## Theory:

Monte Carlo Prediction is a model-free reinforcement learning technique used to estimate value functions directly from experience.

The action-value function is defined as:

\[
Q(s,a) = E[G_t \mid S_t=s, A_t=a]
\]

Where:

- \(Q(s,a)\) = Expected return for taking action \(a\) in state \(s\)
- \(G_t\) = Discounted return after time step \(t\)

Monte Carlo methods estimate action values by averaging returns obtained after visiting each state-action pair over many episodes.

---

## Algorithm:

### Monte Carlo Prediction for Action-Value Function:

**Step-1:** Initialize:
   - Action-value function \(Q(s,a)\)
   - Returns list for every state-action pair

**Step-2:** Generate an episode using a policy.

**Step-3:** For every state-action pair in the episode:
   - Calculate the return \(G\)
   - Store the return for that pair
   - Update \(Q(s,a)\) using the average return

**Step-4:** Repeat the process for many episodes.

**Step-5:** Display the estimated action-value function.

---

## Program:

```python

#Implementation of MC prediction for estimating the action-value function.
import numpy as np
from collections import defaultdict
import gym

env = gym.make("FrozenLake-v1", is_slippery=False)

gamma = 0.9
episodes = 5000

# Action-value function
Q = defaultdict(float)

# Returns storage
returns = defaultdict(list)

# Random policy
def policy(state):
    return env.action_space.sample()

# Generate episode
def generate_episode():

    episode = []

    state, _ = env.reset()

    done = False

    while not done:

        action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        episode.append((state, action, reward))

        state = next_state

    return episode

# Monte Carlo Prediction
for ep in range(episodes):

    episode = generate_episode()

    G = 0

    visited_pairs = set()

    # Traverse backward
    for t in reversed(range(len(episode))):

        state, action, reward = episode[t]

        G = gamma * G + reward

        # First-visit MC
        if (state, action) not in visited_pairs:

            returns[(state, action)].append(G)

            Q[(state, action)] = np.mean(returns[(state, action)])

            visited_pairs.add((state, action))

# Print Q values
print("\nAction Value Function:\n")

for key in Q:
    print(f"State-Action {key}: {Q[key]:.3f}")

```

---

## Output

```text

Action Value Function:

State-Action (8, 1): 0.000
State-Action (4, 1): 0.015
State-Action (4, 0): 0.008
State-Action (0, 1): 0.006
State-Action (0, 0): 0.004
State-Action (0, 3): 0.004
State-Action (4, 3): 0.006
State-Action (1, 0): 0.003
State-Action (0, 2): 0.004
State-Action (4, 2): 0.000
State-Action (8, 3): 0.009
State-Action (1, 1): 0.000
State-Action (1, 3): 0.002
State-Action (2, 0): 0.006
State-Action (2, 3): 0.010
State-Action (1, 2): 0.010
State-Action (10, 2): 0.000
State-Action (6, 1): 0.102
State-Action (2, 1): 0.027
State-Action (9, 3): 0.000
State-Action (10, 0): 0.043
State-Action (14, 3): 0.089
State-Action (13, 2): 0.283
State-Action (9, 1): 0.099
State-Action (8, 2): 0.045
State-Action (8, 0): 0.019
State-Action (9, 0): 0.014
State-Action (3, 1): 0.000
State-Action (3, 3): 0.004
State-Action (2, 2): 0.005
State-Action (3, 0): 0.013
State-Action (6, 2): 0.000
State-Action (10, 3): 0.018
State-Action (9, 2): 0.080
State-Action (6, 0): 0.000
State-Action (13, 3): 0.032
State-Action (13, 1): 0.084
State-Action (14, 0): 0.077
State-Action (10, 1): 0.353
State-Action (13, 0): 0.000
State-Action (3, 2): 0.006
State-Action (6, 3): 0.011
State-Action (14, 2): 1.000
State-Action (14, 1): 0.255

```

---

## Result:

Thus, the Monte Carlo Prediction algorithm was successfully implemented for estimating the action-value function \(Q(s,a)\). The expected returns for different state-action pairs were calculated using sampled episodes generated from the environment.


---
