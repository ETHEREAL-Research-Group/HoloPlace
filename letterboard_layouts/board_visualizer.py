# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# region helpers
x_range = (-0.1397, 0.1397)
y_range = (-0.1078446, 0.1078446)


def visualize_img(path):
  image = mpimg.imread(path)
  # image = mpimg.imread('pink-board.jpeg')

  # Major ticks every 2 units
  x_major_ticks = np.arange(-0.1397, 0.1398, 0.004)
  # Minor ticks every 0.5 units
  y_minor_ticks = np.arange(-0.1078446, 0.1078447, 0.004)

  _, ax = plt.subplots()

  ax.set_xlim(x_range)
  ax.set_ylim(y_range)

  ax.axhline(0, color='black', linewidth=1.0)  # Horizontal line (y = 0)
  ax.axvline(0, color='black', linewidth=1.0)  # Vertical line (x = 0)

  # ax.set_aspect('equal', adjustable='box')
  ax.set_xticks(x_major_ticks)
  ax.set_yticks(y_minor_ticks)

  ax.grid(True, which='major', linestyle='-',
          linewidth=0.8, color='black')  # Major grid
  ax.grid(True, which='minor', linestyle='--', linewidth=0.5,
          color='gray', alpha=0.7)  # Minor grid

  ax.imshow(image, extent=x_range + y_range,
            origin='upper', alpha=0.6, cmap='gray')
  plt.setp(ax.get_xticklabels(), rotation=90)

  plt.show()


def visualize_estimates(positions_letterboard):
  positions_2d = [{'Character': p['Character'], 'x': p['position']
                   [0], 'y': p['position'][2]} for p in positions_letterboard]
  _, ax = plt.subplots()
  ax.set_xlim(x_range)
  ax.set_ylim(y_range)

  x_major_ticks = np.arange(-0.1397, 0.1398, 0.02)
  y_minor_ticks = np.arange(-0.1078446, 0.1078447, 0.02)

  ax.set_xticks(x_major_ticks)
  ax.set_yticks(y_minor_ticks)
  plt.setp(ax.get_xticklabels(), rotation=90)

  # ax.grid(True, linestyle='--', alpha=0.5)

  for pos in positions_2d:
    ax.text(pos['x'], pos['y'], pos['Character'], fontsize=25,
            ha='center', va='center', color='blue')
    # Plot the positions as points


  plt.show()
# endregion


# region Estimates_HardCoded
positions_pinkboard = [
    {'Character': 'A', 'position': [-0.0877, 0, 0.0802]},
    {'Character': 'B', 'position': [-0.0437, 0, 0.0802]},
    {'Character': 'C', 'position': [0.003, 0, 0.0802]},
    {'Character': 'D', 'position': [0.0443, 0, 0.0802]},
    {'Character': 'E', 'position': [0.0843, 0, 0.0802]},

    {'Character': 'F', 'position': [-0.0877, 0, 0.0442]},
    {'Character': 'G', 'position': [-0.0437, 0, 0.0442]},
    {'Character': 'H', 'position': [0.003, 0, 0.0442]},
    {'Character': 'I', 'position': [0.0443, 0, 0.0442]},
    {'Character': 'J', 'position': [0.0843, 0, 0.0442]},

    {'Character': 'K', 'position': [-0.0877, 0, 0.0042]},
    {'Character': 'L', 'position': [-0.0437, 0, 0.0042]},
    {'Character': 'M', 'position': [0.003, 0, 0.0042]},
    {'Character': 'N', 'position': [0.0443, 0, 0.0042]},
    {'Character': 'O', 'position': [0.0843, 0, 0.0042]},

    {'Character': 'P', 'position': [-0.0877, 0, -0.0398]},
    {'Character': 'Q', 'position': [-0.0437, 0, -0.0398]},
    {'Character': 'R', 'position': [0.003, 0, -0.0398]},
    {'Character': 'S', 'position': [0.0443, 0, -0.0398]},
    {'Character': 'T', 'position': [0.0843, 0, -0.0398]},

    {'Character': 'U', 'position': [-0.0877, 0, -0.0758]},
    {'Character': 'V', 'position': [-0.0557, 0, -0.0758]},
    {'Character': 'W', 'position': [-0.0157, 0, -0.0758]},
    {'Character': 'X', 'position': [0.0203, 0, -0.0758]},
    {'Character': 'Y', 'position': [0.0523, 0, -0.0758]},
    {'Character': 'Z', 'position': [0.0843, 0, -0.0758]},
]

positions_alphaboard = [
    {'Character': 'A', 'position': [-0.1197, 0, 0.0882]},
    {'Character': 'B', 'position': [-0.0717, 0, 0.0882]},
    {'Character': 'C', 'position': [-0.0317, 0, 0.0882]},
    {'Character': 'D', 'position': [0.0163, 0, 0.0882]},
    {'Character': 'E', 'position': [0.0603, 0, 0.0882]},
    {'Character': '*', 'position': [0.1063, 0, 0.0922]},
    {'Character': 'F', 'position': [-0.1197, 0, 0.0470]},
    {'Character': 'G', 'position': [-0.0717, 0, 0.0470]},
    {'Character': 'H', 'position': [-0.0317, 0, 0.0470]},
    {'Character': 'I', 'position': [0.0163, 0, 0.0470]},
    {'Character': 'J', 'position': [0.0603, 0, 0.0470]},
    {'Character': ',', 'position': [0.1063, 0, 0.0642]},
    {'Character': 'K', 'position': [-0.1197, 0, 0.0042]},
    {'Character': 'L', 'position': [-0.0717, 0, 0.0042]},
    {'Character': 'M', 'position': [-0.0317, 0, 0.0042]},
    {'Character': 'N', 'position': [0.0163, 0, 0.0042]},
    {'Character': 'O', 'position': [0.0603, 0, 0.0042]},
    {'Character': '!', 'position': [0.1063, 0, 0.0362]},
    {'Character': 'P', 'position': [-0.1197, 0, -0.0358]},
    {'Character': 'Q', 'position': [-0.0717, 0, -0.0358]},
    {'Character': 'R', 'position': [-0.0317, 0, -0.0358]},
    {'Character': 'S', 'position': [0.0163, 0, -0.0358]},
    {'Character': 'T', 'position': [0.0563, 0, -0.0358]},
    {'Character': '?', 'position': [0.1063, 0, -0.0020]},
    {'Character': '.', 'position': [0.1063, 0, -0.0358]},
    {'Character': 'U', 'position': [-0.1197, 0, -0.0758]},
    {'Character': 'V', 'position': [-0.0867, 0, -0.0758]},
    {'Character': 'W', 'position': [-0.0477, 0, -0.0758]},
    {'Character': 'X', 'position': [-0.0077, 0, -0.0758]},
    {'Character': 'Y', 'position': [0.0283, 0, -0.0758]},
    {'Character': 'Z', 'position': [0.0603, 0, -0.0758]},
    {'Character': 'Done', 'position': [0.1043, 0, -0.0798]},
]
# endregion

if __name__ == '__main__':
  visualize_img('alpha-letterboard.jpg')
  visualize_estimates(positions_alphaboard)

  visualize_img('pink-board.jpeg')
  visualize_estimates(positions_pinkboard)