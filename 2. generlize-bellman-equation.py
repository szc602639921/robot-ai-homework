# %%
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- #
#       Plotting Utilities      #
# ----------------------------- #

def plot_environment(quantity, n_x, n_y, origin='upper', figsize=(4,3), annotate=True, fontsize=20):
    """
    Visualize a scalar quantity (like rewards or values) on a grid.
    """
    plt.figure(figsize=figsize)
    plt.imshow(quantity, origin=origin, cmap='coolwarm', vmin=-1, vmax=1)
    
    ax = plt.gca()
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2)
    
    ax.set_xticks(np.arange(n_x + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_y + 1) - 0.5, minor=True)
    plt.xticks(range(n_x))
    plt.yticks(range(n_y))
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(n_y):
            for j in range(n_x):
                plt.text(j, i, round(quantity[i, j], 3), fontsize=fontsize,
                         ha="center", va="center", color="white")

    return plt


def plot_policy(quantity, n_x, n_y, origin='upper', figsize=(4,3), annotate=True, fontsize=20):
    """
    Visualize policy directions as arrows on a grid.
    """
    plt.figure(figsize=figsize)
    plt.imshow(np.zeros_like(quantity), origin=origin, cmap='coolwarm', vmin=-1, vmax=1)  # blank background
    
    ax = plt.gca()
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(4)

    ax.set_xticks(np.arange(n_x + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_y + 1) - 0.5, minor=True)
    plt.xticks(range(n_x), fontsize=20)
    plt.yticks(range(n_y), fontsize=20)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Direction encoding: (dx, dy)
    directions = {
        0: (0, 0.3),   # Up (visually downward)
        1: (0.3, 0),   # Right
        2: (0, -0.3),  # Down (visually upward)
        3: (-0.3, 0),  # Left
    }

    if annotate:
        for i in range(n_y):
            for j in range(n_x):
                dir_code = quantity[n_y - i -1, j]
                if dir_code in directions:
                    dx, dy = directions[dir_code]
                    plt.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc='k', ec='k')

    if origin == 'upper':
        ax.invert_yaxis()

    plt.tight_layout()
    return plt

# ----------------------------- #
#       MDP Setup               #
# ----------------------------- #

# Grid dimensions
n_x, n_y = 6, 4
gamma = 0.9

# Terminal reward locations
location_positive_reward = (0, 5)
location_negative_reward = (3, 5)

# Random policy direction grid: 0=up, 1=right, 2=down, 3=left
initial_policy_grid = np.random.randint(0, 4, size=(n_y, n_x))

def policy(state, grid, pos_reward, neg_reward):
    """
    Given a state and a grid of directions, return the next state
    according to the policy (or stay in place if terminal or out-of-bounds).
    """
    y, x = state
    direction = grid[y, x]

    if state in [pos_reward, neg_reward]:
        return direction, state

    # Movement mapping
    moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    inverse = {0: 2, 1: 3, 2: 0, 3: 1}

    dy, dx = moves[direction]
    next_y, next_x = y + dy, x + dx
    max_y, max_x = grid.shape

    # Reverse direction if out of bounds
    if not (0 <= next_y < max_y and 0 <= next_x < max_x):
        direction = inverse[direction]
        dy, dx = moves[direction]
        next_y, next_x = y + dy, x + dx
        if not (0 <= next_y < max_y and 0 <= next_x < max_x):
            return direction, (y, x)

    return direction, (next_y, next_x)


def compute_policy_grid(grid):
    """
    For each cell, compute the updated policy direction considering boundaries and terminals.
    """
    updated_grid = np.zeros_like(grid)
    for i in range(n_y):
        for j in range(n_x):
            updated_grid[i, j], _ = policy((i, j), grid, location_positive_reward, location_negative_reward)
    return updated_grid

# Build policy grid and plot it
policy_grid = compute_policy_grid(initial_policy_grid)
plot_policy(policy_grid, n_x, n_y).show()

# ----------------------------- #
#       Reward Field            #
# ----------------------------- #

def rewards(state, pos_reward, neg_reward):
    if state == pos_reward:
        return 1
    elif state == neg_reward:
        return -1
    return 0

# Build reward map
reward_grid = np.zeros_like(policy_grid, dtype=float)
for i in range(n_y):
    for j in range(n_x):
        reward_grid[i, j] = rewards((i, j), location_positive_reward, location_negative_reward)

#plot_environment(reward_grid, n_x, n_y, annotate=False).show()

# ----------------------------- #
#       Policy Evaluation       #
# ----------------------------- #

def map_state_to_flat_index(state, n_x):
    y, x = state
    return y * n_x + x

def map_flat_index_to_state(index, n_x):
    return divmod(index, n_x)

# Right-hand side vector: immediate rewards
b = np.zeros(n_x * n_y)
for i in range(n_y):
    for j in range(n_x):
        idx = map_state_to_flat_index((i, j), n_x)
        b[idx] = rewards((i, j), location_positive_reward, location_negative_reward)

# Left-hand side matrix: identity minus gamma * transition
A = np.eye(n_x * n_y)
for i in range(n_x * n_y):
    state = map_flat_index_to_state(i, n_x)
    _, next_state = policy(state, policy_grid, location_positive_reward, location_negative_reward)
    j = map_state_to_flat_index(next_state, n_x)

    # Skip terminal states (self-loop)
    if next_state != state:
        A[i, j] -= gamma

# Solve system of linear equations
value_function = np.linalg.solve(A, b)
value_function = value_function.reshape((n_y, n_x))

# Plot value function
plot_environment(value_function, n_x, n_y, figsize=(10, 5)).show()
# %%
value_function = np.linalg.solve(A, b)
# %%
value_function.shape
# %%
value_function
# %%
def evaluate_policy_bellman(value_function, rewards_field, policy, discount=0.9, theta=1e-5):
    n_y, n_x = value_function.shape
    while True:
        delta = 0
        new_v = np.copy(value_function)
        for y in range(n_y):
            for x in range(n_x):
                _, (next_y, next_x) = policy((y, x), policy_grid, location_positive_reward, location_negative_reward)
                r = rewards_field[y, x]
                v_next = value_function[next_y, next_x]
                new_v[y, x] = r + discount * v_next
                delta = max(delta, abs(new_v[y, x] - value_function[y, x]))
        value_function = new_v
        if delta < theta:
            break
    return value_function/np.max(value_function)
# %%
value_function_bellman = evaluate_policy_bellman(np.zeros_like(reward_grid), reward_grid, policy, 0.9)
plot_environment(value_function_bellman, n_x, n_y,figsize=(10, 5),annotate=True)
# %%
