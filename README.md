# Repository for ACM IUI '24 Paper Submission

This is the corresponding repository for the paper published in ACM IUI24: [https://doi.org/10.1145/3640543.3645153]. Please cite the paper if you intend to use the data or the code.

This README outlines the structure and details of our dataset.

**Scenario IDs:**
- Scenario 1: `1c54d7`
- Scenario 2: `b3f9f8`
- Scenario 3: `40ad1b`

- **Participant 1 ID:** `a19cfd`
- **Participant 2 ID:** `8e5234`

**Participant IDs for the updated app with the gem game at the end and lowered threshold for data collection**
- **Participant 3 ID** `8d418f` Colleced about 50% data but subject did not test the system at the end
- **Participant 4 ID** `9ab3fe` Collected 100% data and subject tested out the gems at the end
- **Participant 5 ID** `e75dd7` Collected 100% data and subject tested out the gems at the end and attempted spelling
- **Participant 6 ID** `347193` Collected 100% data and subject tested out the gems at the end
- **Participant 7 ID** `0edbb4` Collected ~15% data and subject tested out the gems at the end 


> **Note:** In all scenarios, eye gaze data has been replaced by head gaze data. We disabled eye tracking to eliminate the need for the calibration process.

---

## File Structure

### `data.csv` 

This file contains various parameters tracked during our experiments. The following columns are included:

- `timestamp`: Timestamp of the tracked data
- `cam_pos`: Proxy for the head's position
- `cam_rot`: Head rotation
- `rif_pos`: Right index finger position
- `rif_rot`: Right index finger rotation
- `rpa_pos`: Right palm position
- `rpa_rot`: Right palm rotation
- `act_pos`: Intended for tracking target movement, currently unused due to observed issues
- `act_rot`: Similar to `act_pos`, currently unused

### `all_data.csv`

This file contains a more comprehensive set of tracked data. Although not currently in use, it was collected for potential future work. It includes:

- `timestamp`: Timestamp of the tracked data
- `rif_pos`, `rif_rot`, `rif_vel`, `rif_ang_vel`: Right index finger's position, rotation, velocity, and angular velocity respectively
- `lif_pos`, `lif_rot`, `lif_vel`, `lif_ang_vel`: Left index finger's position, rotation, velocity, and angular velocity respectively
- `rPalm_pos`, `rPalm_rot`, `rPalm_vel`, `rPalm_ang_vel`: Right palm's position, rotation, velocity, and angular velocity respectively
- `lPalm_pos`, `lPalm_rot`, `lPalm_vel`, `lPalm_ang_vel`: Left palm's position, rotation, velocity, and angular velocity respectively
- `cam_pos`, `cam_rot`, `cam_vel`, `cam_ang_vel`: Camera (Head) position, rotation, velocity, and angular velocity respectively
- `tar_pos`, `tar_rot`, `tar_vel`, `tar_ang_vel`: Target (Letterboard) position, rotation, velocity, and angular velocity respectively
- `Is Eye Tracking Enabled and Valid`: A status flag for eye tracking
- `gaze origin`, `gaze direction`, `gaze rotation`: Gaze metrics
- `head movement direction`, `head velocity`: Head movement metrics
