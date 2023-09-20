# Repository for ACM SUI '23 Paper Submission

Welcome to the repository for our paper submission to ACM SUI '23. This README outlines the structure and details of our dataset.

**Scenario IDs:**
- Scenario 1: `1c54d7`
- Scenario 2: `b3f9f8`
- Scenario 3: `40ad1b`

- **Participant 1 ID:** `a19cfd`
- **Participant 2 ID:** `8e5234`
- **Participant 3 ID:** `a4cc93` For this subject, the collision data was not detected due to system error.
- **Participant 4 ID:** `d79960` For this subject, the collision data was not detected due to system error.

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
