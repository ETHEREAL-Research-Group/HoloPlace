# %%
# region Imports
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from ast import literal_eval
from scipy.spatial.transform import Rotation as R
import os
# endregion


# region Helpers
def extract_position_rotation(transform):
  position = transform[:3, 3]
  rotation_matrix = transform[:3, :3]
  rotation = R.from_matrix(rotation_matrix).as_quat()

  return position, rotation


def get_xz_dist(entry1, entry2):
      return np.linalg.norm(np.array([entry1[0], entry1[2]]) - np.array([entry2[0], entry2[2]]))


def world_to_local_matrix(position, quaternion):
  quaternion = normalize_quaternion(quaternion)
  rotation_matrix = R.from_quat(quaternion).as_matrix().T

  translation = -np.dot(rotation_matrix, position)

  inverse_transform = np.eye(4)
  inverse_transform[:3, :3] = rotation_matrix
  inverse_transform[:3, 3] = translation

  return inverse_transform


def local_to_world_matrix(position, quaternion):
  rotation_matrix = R.from_quat(quaternion).as_matrix()

  transform = np.eye(4)  # Initialize as identity matrix
  transform[:3, :3] = rotation_matrix
  transform[:3, 3] = position  # Top-right 3x1 part is the translation vector

  return transform


def position_distance(point1, point2):
  position1 = point1[:3]
  position2 = point2[:3]
  return euclidean(position1, position2) * 100


epsilon = np.finfo(float).eps


def normalize_quaternion(quaternion):
  quaternion_norm = np.linalg.norm(quaternion)
  normalized = quaternion / (quaternion_norm+epsilon)
  return normalized


def geodesic_loss(point1, point2):
  q1 = normalize_quaternion(point1[-4:])
  q2 = normalize_quaternion(point2[-4:])
  dot_product = np.dot(q1, q2)
  angle = np.abs(np.arccos(2 * dot_product**2 - 1))
  return np.rad2deg(angle)


def matrix_dot(T1, T2):
  return np.dot(T1, T2)


def matrix_inverse(T):
  return np.linalg.inv(T)
# endregion


# region Letterboard_positions

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


# endregion


# region Touch_entries
def get_selection_accuracy(user_id, _letterboard_positions):
  letterboard_positions = pd.DataFrame(_letterboard_positions)

  data_df = pd.read_csv(f'data/{user_id}/data.csv')
  all_data_df = pd.read_csv(f'data/{user_id}/all_data.csv')
  events_df = pd.read_csv(f'data/{user_id}/events.csv')
  true_data = np.load(f'data/{user_id}/output/true.npy')
  pred_data = np.load(f'data/{user_id}/output/pred.npy')

  right_index_df = events_df[events_df["event"]
                                     == 'Right IndexTip'].reset_index(drop=True)
  rows_to_drop = []
  for i in range(1, len(right_index_df)):
    if right_index_df.loc[i, "timestamp"] - right_index_df.loc[i - 1, "timestamp"] < 350:
      rows_to_drop.append(i)

  # Drop the rows
  right_index_df_cleaned = right_index_df.drop(rows_to_drop).reset_index(drop=True)

  right_index_timestamps = right_index_df_cleaned["timestamp"].values.astype(int)

  

  right_index_timestamps.sort()
  # print(right_index_timestamps)
  data_df["timestamp"] = data_df["timestamp"].astype(int)
  all_data_df["timestamp"] = all_data_df["timestamp"].astype(int)
  data_df.dropna(inplace=True)
  data_df.reset_index(inplace=True)
  all_data_df = all_data_df[all_data_df['rif_pos'].notna()]
  all_data_df.reset_index(inplace=True)
  data_df["cam_pos"] = data_df["cam_pos"].map(literal_eval, na_action='ignore')
  data_df["cam_rot"] = data_df["cam_rot"].map(literal_eval, na_action='ignore')
  data_df["rif_pos"] = data_df["rif_pos"].map(literal_eval, na_action='ignore')
  data_df["rif_rot"] = data_df["rif_rot"].map(literal_eval, na_action='ignore')
  data_df["cam_full"] = data_df.apply(
      lambda row: np.concatenate([row["cam_pos"], row["cam_rot"]]), axis=1)

  data_df['rif_pos_y'] = data_df['rif_pos'].apply(lambda pos: pos[1])
  data_df['rif_pos_y_velocity'] = data_df['rif_pos_y'].diff() / data_df['timestamp'].diff()
  upper_boundary = 0.01
  lower_boundary = 0.005
  # Iterate through the rows
  collision_timestamps = []
  state = None
  for i in range(len(data_df)):
    y = data_df.loc[i, 'rif_pos_y']
    timestamp = data_df.loc[i, 'timestamp']
    
    if state is None and y > upper_boundary:
      # Transition to "above" state
      state = "above"
    
    elif state == "above" and y < lower_boundary:
      # Collision detected when transitioning from "above" to "below"
      collision_timestamps.append(timestamp)
      state = None  # Reset state after detecting a collision
  # data_df['above_upper'] = data_df['rif_pos_y'] > upper_boundary
  # data_df['below_lower'] = data_df['rif_pos_y'] < lower_boundary
  # collisions = data_df[(data_df['above_upper'].shift(1) == True) & (data_df['below_lower'] == True)]
  # # collisions = collisions[collisions['rif_pos_y_velocity'] < 0]
  # collision_timestamps = collisions['timestamp'].tolist()
  # collision_timestamps.sort()
  # print(f'collision_ts_list = {collision_timestamps}')

  # threshold = 0.005  # cm
  # data_df['rif_pos_y_sign'] = data_df['rif_pos_y'] > threshold  # True if positive, False otherwise
  # transitions = data_df[(data_df['rif_pos_y_sign'].shift(1) == True) & (data_df['rif_pos_y_sign'] == False)]

  # transition_timestamps = transitions['timestamp'].tolist()
  # transition_timestamps.sort()
  # raise Exception()

  all_data_df["rif_pos"] = all_data_df["rif_pos"].map(
      literal_eval, na_action='ignore')
  all_data_df["rif_rot"] = all_data_df["rif_rot"].map(
      literal_eval, na_action='ignore')
  all_data_df["cam_pos"] = all_data_df["cam_pos"].map(
      literal_eval, na_action='ignore')
  all_data_df["cam_rot"] = all_data_df["cam_rot"].map(
      literal_eval, na_action='ignore')
  all_data_df["tar_pos"] = all_data_df["tar_pos"].map(
      literal_eval, na_action='ignore')
  all_data_df["tar_rot"] = all_data_df["tar_rot"].map(
      literal_eval, na_action='ignore')

  selection_accuracy = []

  for ts in right_index_timestamps:
  # for ts in collision_timestamps:
    #region TEMP
    # # entry = data_df[data_df['timestamp'] == ts].iloc[0]
    # time_differences = (data_df["timestamp"] - ts).abs()

    # closest_idx = time_differences.idxmin()

    # closest_timestamp = data_df.loc[closest_idx, "timestamp"]
    # entry = data_df.iloc[closest_idx]
    # distances = []
    # for _, letter in letterboard_positions.iterrows():
    #   letter_pos = np.array(letter["position"])
    #   distance = get_xz_dist(entry['rif_pos'], letter_pos)
    #   distances.append((letter["Character"], distance, entry['rif_pos'][1]))

    # closest_letter = min(distances, key=lambda x: x[1])
    # print(
    #     f"Closest letter {closest_letter[0]} with distance {closest_letter[1]:.4f}, y of hand is {closest_letter[2]}")
    # print('-'*100)
    # continue
    #endregion
    time_differences = (data_df["timestamp"] - ts).abs()

    closest_idx = time_differences.idxmin()

    closest_timestamp = data_df.loc[closest_idx, "timestamp"]
    closest_value = data_df.loc[closest_idx, "cam_full"]
    ts_difference_1 = abs(closest_timestamp - ts)
    if ts_difference_1 >= 50:
      raise Exception()
    temp = [true_idx for true_idx, true_entry in enumerate(
        true_data) if euclidean(closest_value, true_entry) < 1e-5]
    if len(temp) > 1:
      print('WARNING!')
      continue
      # raise Exception()
    if len(temp) == 0:
      continue

    time_differences2 = (all_data_df["timestamp"] - ts).abs()
    closest_idx2 = time_differences2.idxmin()

    true_idx = temp[0]
    true_entry = true_data[true_idx]
    pred_entry = pred_data[true_idx]
    data_entry = data_df.iloc[closest_idx]
    all_data_entry = all_data_df.iloc[closest_idx2]

    ts_difference_2 = all_data_entry['timestamp'] - ts

    if ts_difference_2 > 50:
      raise Exception()

    target_local_to_world = local_to_world_matrix(
        all_data_entry['tar_pos'], all_data_entry['tar_rot'])
    finger_in_target_local = local_to_world_matrix(
        data_entry['rif_pos'], data_entry['rif_rot'])
    finger_local_to_world = matrix_dot(
        target_local_to_world, finger_in_target_local)
    predicted_camera_local_to_world = local_to_world_matrix(
        pred_entry[:3], pred_entry[3:])
    camera_local_to_world = local_to_world_matrix(
        all_data_entry['cam_pos'], all_data_entry['cam_rot'])
    predicted_target_local_to_world = matrix_dot(
        camera_local_to_world, matrix_inverse(predicted_camera_local_to_world))

    finger_pos_calculate, finger_rot_calculated = extract_position_rotation(
        finger_local_to_world)
    predicted_tar_pos, predicted_tar_rot = extract_position_rotation(
        predicted_target_local_to_world)

    finger_in_predicted_target_local = matrix_dot(matrix_inverse(
        predicted_target_local_to_world), finger_local_to_world)
    finger_pos_in_predicted_target_local, finger_rot_in_predicted_target_local = extract_position_rotation(
        finger_in_predicted_target_local)

    # print(f"finger world = {all_data_entry['rif_pos']}")
    # print(f"finger in target local = {data_entry['rif_pos']}")
    # print(f"finger in predicted target local = {finger_pos_in_predicted_target_local}")
    # print(f"finger in world but calculated = {finger_pos_calculate}")
    # print(f"tar pos = {all_data_entry['tar_pos']}")
    # print(f"predicted tar pos = {predicted_tar_pos}")
    # print(f"pos error from before = {position_distance(true_entry, pred_entry):.2f}")
    # print(f"new pos error = {position_distance(all_data_entry['tar_pos'], predicted_tar_pos):.2f}")

    distances = []
    distances_pred = []
    
    for _, letter in letterboard_positions.iterrows():
      letter_pos = np.array(letter["position"])
      distance = get_xz_dist(data_entry['rif_pos'], letter_pos)
      distances.append((letter["Character"], distance, data_entry['rif_pos'][1]))
      distance_pred = get_xz_dist(
          finger_pos_in_predicted_target_local, letter_pos)
      distances_pred.append((letter["Character"], distance_pred, data_entry['rif_pos'][1]))

    closest_letter = min(distances, key=lambda x: x[1])
    closest_letter_pred = min(distances_pred, key=lambda x: x[1])
    if closest_letter_pred[0] != closest_letter[0]:
      print(
          f"Closest letter {closest_letter[0]} with distance {closest_letter[1]:.4f}, y of hand is {closest_letter[2]}")
      print(
          f"Closest letter pred {closest_letter_pred[0]} with distance {closest_letter_pred[1]:.4f}, y of hand is {closest_letter_pred[2]}")
      print('*'*100)
    selection_accuracy.append(closest_letter_pred[0] == closest_letter[0])

    # print(f'rot_err = {geodesic_loss(true_entry, pred_entry):.2f}, pos error = {position_distance(true_entry, pred_entry):.2f}')
  selection_accuracy_percentage = selection_accuracy.count(True)/len(selection_accuracy)
  print(f'Target selection accuracy is = {selection_accuracy_percentage}. Total number of transaction = {len(selection_accuracy)}')
  return selection_accuracy_percentage, len(selection_accuracy)
# endregion


base_path = './data'
dir_list = os.listdir(base_path)
# dir_list = ['ahmad']
# dir_list = ['6167a0']
selction_accuracies = []
print(dir_list)
user_board = {
    '1c54d7': positions_pinkboard,  # scenario 1
    'b3f9f8': positions_pinkboard,  # scenario 2
    '40ad1b': positions_pinkboard,  # scenario 3
    'a19cfd': positions_pinkboard,  # [NOT SURE] participant 1
    '8e5234': positions_pinkboard,  # participant 2
    '8d418f': positions_alphaboard,  # participant 3
    '9ab3fe': positions_alphaboard,  # participant 4
    'e75dd7': positions_alphaboard,  # participant 5
    '347193': positions_alphaboard,  # participant 6
    '0edbb4': positions_pinkboard,  # participant 7
    '6167a0': positions_alphaboard,  # [NOT SURE] participant 8
    # 'ahmad': positions_pinkboard,
}

participant_map = {
  '1c54d7': 'Scenario1',
  'b3f9f8': 'Scenario2',
  '40ad1b': 'Scenario3',
  'a19cfd': 'P1',
  '8e5234': 'P2',
  '8d418f': 'P3',
  '9ab3fe': 'P4',
  'e75dd7': 'P5',
  '347193': 'P6',
  '0edbb4': 'P7',
  '6167a0': 'P8',
}

for user_id in user_board.keys():
  print(f'processing {participant_map[user_id]}...')
  accuracy, no_interaction = get_selection_accuracy(user_id, user_board[user_id])
  selction_accuracies.append(
      {'user': participant_map[user_id], 'accuracy': accuracy, 'num_interactions': no_interaction})

  print('-'*200)

for item in selction_accuracies:
  print(item)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import spearmanr, linregress, pearsonr
from pymannkendall import original_test  # Install with pip install pymannkendall
import matplotlib.pyplot as plt

results_df = pd.read_csv('Results.csv')

t_stat, p_value = ttest_rel(results_df['Data'], results_df['Selection Accuracy (percentage)'])

print(f"Paired t-test result: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}, {len(results_df['Data'].dropna())}")

correlation, p_value = pearsonr(results_df['Data'], results_df['Selection Accuracy (percentage)'])
print("Pearson correlation:", correlation)
print("P-value:", p_value)

results_df = results_df.sort_values(by="Data")

x = results_df["Data"]
y = results_df["Selection Accuracy (percentage)"]
# 1. Spearman Correlation
spearman_corr, spearman_pval = spearmanr(x, y)

# 2. Linear Regression
linreg = linregress(x, y)

# 3. Mann-Kendall Test
mk_test = original_test(y)

# Print statistical results
print(f"Spearman Correlation: {spearman_corr:.2f}, p-value: {spearman_pval:.4f}")
print(f"Linear Regression Slope: {linreg.slope:.4f}, Intercept: {linreg.intercept:.2f}, p-value: {linreg.pvalue:.4f}")
print(f"Mann-Kendall Test: Trend = {mk_test.trend}, p-value = {mk_test.p:.4f}")

# Plotting with Linear Regression Line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue")
participants = results_df["Participant"].unique()
colors = plt.cm.tab10.colors  # Use a colormap for colors
markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>', '.']  # Define marker styles

for i, participant in enumerate(participants):
  participant_data = results_df[results_df["Participant"] == participant]
  plt.scatter(
      participant_data["Data"], 
      participant_data["Selection Accuracy (percentage)"],
      label=f"{participant}",
      color=colors[i % len(colors)],
      marker=markers[i % len(markers)],
      s=100,  # Size of the marker
      edgecolor='black'
  )

# First legend for participants (scatter points)
scatter_legend = plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.gca().add_artist(scatter_legend)  # Add the first legend manually to the plot

plt.plot(x, linreg.intercept + linreg.slope * x, color="red", label="Linear Regression Line")
plt.xlabel("Number of Data Samples")
plt.ylabel("Selection Accuracy (%)")

# Highlight the target threshold
target_accuracy = 80  # Define the target accuracy
threshold_data = results_df[results_df["Selection Accuracy (percentage)"] >= target_accuracy]["Data"].min()

plt.axhline(y=target_accuracy, color="green", linestyle="--", label=f"Target Accuracy: {target_accuracy}%")
plt.axvline(x=7481, color="orange", linestyle="--", label=f"Threshold Data: {7481}")

# Second legend for the regression line and other elements
handles, labels = plt.gca().get_legend_handles_labels()
line_legend = plt.legend(handles[-3:], labels[-3:], loc="upper left", bbox_to_anchor=(1.05, 0.2), borderaxespad=0.)

plt.grid(True)
plt.tight_layout()
plt.savefig(f'plots/selection_accuracy.png', transparent=True)
plt.show()

#%%
plt.figure(figsize=(10, 6))
plt.plot(x, results_df["Positional Error (cm)"], color="blue")
participants = results_df["Participant"].unique()
colors = plt.cm.tab10.colors  # Use a colormap for colors
markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>', '.']  # Define marker styles

for i, participant in enumerate(participants):
  participant_data = results_df[results_df["Participant"] == participant]
  plt.scatter(
      participant_data["Data"], 
      participant_data["Positional Error (cm)"],
      label=f"{participant}",
      color=colors[i % len(colors)],
      marker=markers[i % len(markers)],
      s=100,  # Size of the marker
      edgecolor='black'
  )
scatter_legend = plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.gca().add_artist(scatter_legend)  # Add the first legend manually to the plot

plt.axvline(x=7481, color="orange", linestyle="--", label=f"Threshold Data: {7481}")
handles, labels = plt.gca().get_legend_handles_labels()
line_legend = plt.legend(handles[-1:], labels[-1:], loc="upper left", bbox_to_anchor=(1.05, 0.2), borderaxespad=0.)
plt.xlabel("Number of Data Samples")
plt.ylabel("Positional Error (cm)")

plt.grid(True)
plt.tight_layout()
plt.savefig(f'plots/position_error_data.png', transparent=True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, results_df["Rotational Error (deg)"], color="blue")
participants = results_df["Participant"].unique()
colors = plt.cm.tab10.colors  # Use a colormap for colors
markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>', '.']  # Define marker styles

for i, participant in enumerate(participants):
  participant_data = results_df[results_df["Participant"] == participant]
  plt.scatter(
      participant_data["Data"], 
      participant_data["Rotational Error (deg)"],
      label=f"{participant}",
      color=colors[i % len(colors)],
      marker=markers[i % len(markers)],
      s=100,  # Size of the marker
      edgecolor='black'
  )
scatter_legend = plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.gca().add_artist(scatter_legend)  # Add the first legend manually to the plot

plt.axvline(x=7481, color="orange", linestyle="--", label=f"Threshold Data: {7481}")
handles, labels = plt.gca().get_legend_handles_labels()
line_legend = plt.legend(handles[-1:], labels[-1:], loc="upper left", bbox_to_anchor=(1.05, 0.2), borderaxespad=0.)
plt.xlabel("Number of Data Samples")
plt.ylabel("Rotational Error (deg)")

plt.grid(True)
plt.tight_layout()
plt.savefig(f'plots/rotation_error_data.png', transparent=True)

plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

data = pd.read_csv('camera_before_after.csv')
# Data

labels = data['Participant'].values
before_position_mean = data['Before Camera Position Mean'].values * 100 
before_position_sd = data['Before Camera Position SD'].values * 100 
before_rotation_mean = data['Before Camera Rotation Mean'].values * (180.0/np.pi) 
before_rotation_sd = data['Before Camera Rotation Mean'].values * (180.0/np.pi)

after_position_mean = data['After Camera Position Mean'].values * 100 
after_position_sd = data['After Camera Position SD'].values * 100 
after_rotation_mean = data['After Camera Rotation Mean'].values * (180.0/np.pi) 
after_rotation_sd = data['After Camera Rotation Mean'].values * (180.0/np.pi)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('camera_before_after.csv')

# Extract data
labels = data['Participant'].values
x = np.arange(len(labels))
# Position metrics
before_position_mean = data['Before Camera Position Mean'].values * 100 
before_position_sd = data['Before Camera Position SD'].values * 100 
after_position_mean = data['After Camera Position Mean'].values * 100 
after_position_sd = data['After Camera Position SD'].values * 100 

# Rotation metrics
before_rotation_mean = data['Before Camera Rotation Mean'].values * (180.0 / np.pi) 
before_rotation_sd = data['Before Camera Rotation SD'].values * (180.0 / np.pi) 
after_rotation_mean = data['After Camera Rotation Mean'].values * (180.0 / np.pi) 
after_rotation_sd = data['After Camera Rotation SD'].values * (180.0 / np.pi)

mpl.rcParams['font.size'] = 16

# Plot position (mean with error bars)
plt.figure(figsize=(10, 6))
plt.errorbar(x - 0.2, before_position_mean, yerr=before_position_sd, fmt='o', label='Before Transformation', capsize=5, color='blue')
plt.errorbar(x + 0.2, after_position_mean, yerr=after_position_sd, fmt='o', label='After Transformation', capsize=5, color='orange')
plt.xlabel('Experiment')
plt.ylabel('Position Mean and SD (cm)')
plt.legend()
plt.xticks(x, labels, rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'plots/camera_before_after.png', transparent=True)
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(x - 0.1, before_rotation_sd, color='blue', label='Before', width=0.1, alpha=0.7)
plt.bar(x + 0.1, after_rotation_sd, color='orange', label='After', width=0.1, alpha=0.7)
plt.ylabel('Rotation SD (degrees)')
plt.xlabel('Experiment')
plt.xticks(rotation=45)
plt.xticks(x, labels, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'plots/camera_before_after_rotation.png', transparent=True)
plt.show()

from scipy.stats import ttest_rel

stat, p_value = ttest_rel(before_rotation_sd, after_rotation_sd)
print(f"Paired t-test: t = {stat:.2f}, p = {p_value:.4f}")

#%%
