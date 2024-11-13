# Observation Space

The observation space consists of the following parts (in order)

## The joints

| joint      | id |
| ---------- | --- |
| l_sho_pitch       | 0      |
| l_sho_roll       | 1      |
| l_el       | 2      |
| r_sho_pitch     | 3      |
| r_sho_roll     | 4      |
| r_el     | 5      |
| l_hip_yaw      | 6      |
| l_hip_roll      | 7      |
| l_hip_pitch      | 8      |
| l_knee      | 9      |
| l_ank_pitch       | 10     |
| l_ank_roll | 11     |
| r_hip_yaw | 12     |
| r_hip_roll     | 13     |
| r_hip_pitch  | 14     |
| r_knee  | 15     |
| r_ank_pitch     | 16     |
| r_ank_roll     | 17     |
| head_pan     | 18     |
| head_tilt     | 19     |

> 18 and 19 aren't used in current research

## The body parts

| body part       | id  |
| --------------- | --- |
| body_link           | 0  |
| head_pan_link          | 1  |
| head_tilt_link          | 2  |
| l_sho_pitch_link     | 3  |
| l_sho_roll_link       | 4  |
| l_el_link      | 5  |
| r_sho_pitch_link      | 6  |
| r_sho_roll_link        | 7  |
| r_el_link       | 8  |
| l_hip_yaw_link | 9  |
| l_hip_roll_link | 10 |
| l_hip_pitch_link  | 11 |
| l_knee_link  | 12 |
| l_ank_pitch_link  | 13 |
| l_ank_roll_link  | 14 |
| r_hip_yaw_link  | 15 |
| r_hip_roll_link  | 16 |
| r_hip_pitch_link  | 17 |
| r_ank_pitch_link  | 18 |
| r_ank_roll_link  | 19 |

## The Action Space

Min: -3.14
Max: 3.14
Unit: rad

| Action       | Actuator | Joint |
| ------------ | -------- | ----- |
| Left Shoulder Pitch | l_sho_pitch_act | l_sho_pitch |
| Left Shoulder Roll | l_sho_roll_act | l_sho_roll |
| Left Elbow | l_el_act | l_el |
| Right Shoulder Pitch | r_sho_pitch_act | r_sho_pitch |
| Right Shoulder Roll | r_sho_roll_act | r_sho_roll |
| Right Elbow | r_el_act | r_el |
| Left Hip Yaw | l_hip_yaw_act | l_hip_yaw |
| Left Hip Roll | l_hip_roll_act | l_hip_roll |
| Left Hip Pitch | l_hip_pitch_act | l_hip_pitch |
| Left Knee | l_knee_act | l_knee |
| Left Ankle Pitch | l_ank_pitch_act | l_ank_pitch |
| Left Ankle Roll | l_ank_roll_act | l_ank_roll |
| Right Hip Yaw | r_hip_yaw_act | r_hip_yaw |
| Right Hip Roll | r_hip_roll_act | r_hip_roll |
| Right Hip Pitch | r_hip_pitch_act | r_hip_pitch |
| Right Knee | r_knee_act | r_knee |
| Right Ankle Pitch | r_ank_pitch_act | r_ank_pitch |
| Right Ankle Roll | r_ank_roll_act | r_ank_roll |

## The Observation Space
