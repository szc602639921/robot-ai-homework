# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
def plot_environment(quantity, n_x, n_y, origin='upper', figsize=(4,3), annotate=True, fontsize=20):

    plt.figure(figsize=figsize)
    plt.imshow(quantity, origin=origin, cmap='coolwarm', vmin=-1, vmax=1)
    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(2)
    plt.gca().set_xticks(np.arange(n_x + 1) - .5, minor=True)
    plt.gca().set_yticks(np.arange(n_y + 1) - .5, minor=True)
    plt.xticks([i for i in range(0,n_x)])
    plt.yticks([i for i in range(0,n_y)])
    plt.gca().grid(which="minor", color="black", linewidth=1)
    plt.gca().tick_params(which="minor", bottom=False, left=False)
    if annotate:
        for i in range(n_y):
            for j in range(n_x):
                text = plt.text(j, i, round(quantity[i, j], 3), fontsize=fontsize, ha="center", va="center", color="w")
    
    return plt

# %%
discount = 0.9
n_width = 3
n_height = 2
n_states = n_width * n_height
# %%
environment = np.zeros((n_height, n_width))
# %%
plt = plot_environment(environment, n_width, n_height, annotate=False)
plt.savefig('small-RL-environement.pdf', dpi=300, bbox_inches='tight')
# %%
rewards_field = np.zeros_like(environment)
rewards_field[0,2] = 1
# %%
plt = plot_environment(rewards_field, n_width, n_height, annotate=True)
plt.savefig('small-RL-environement-rewards-field.pdf', dpi=300, bbox_inches='tight')
# %%
# solve the bellman equations and plot the values in the cells

# %%
