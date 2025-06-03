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
plot_policy(initial_policy_grid, n_x, n_y).show()
# %%
def policy(state, grid, pos_reward, neg_reward):
    #TODO

    return None


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

plot_environment(reward_grid, n_x, n_y, annotate=False).show()
# %%
# ----------------------------- #
#       Policy Evaluation       #
# ----------------------------- #

def map_state_to_flat_index(state, n_x):
    y, x = state
    return y * n_x + x

def map_flat_index_to_state(index, n_x):
    return divmod(index, n_x)

# TODO Right-hand side vector: immediate rewards


# TODO Left-hand side matrix: identity minus gamma * transition


# Solve system of linear equations
value_function = np.linalg.solve(A, b)
value_function = value_function.reshape((n_y, n_x))

# Plot value function
plot_environment(value_function, n_x, n_y, figsize=(10, 5)).show()
# %%
