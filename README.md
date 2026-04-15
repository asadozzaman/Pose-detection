# Pose Detection & Activity Classification

Real-time pose estimation and rule-based activity classification on video using YOLOv8 Pose. Detects multiple people simultaneously, draws their skeletons, and labels each person's activity per frame.

---

## Features

- Multi-person pose estimation via YOLOv8 Pose
- Rule-based activity classification from 17 COCO keypoints
- Per-person activity label smoothing using a majority-vote sliding window
- Color-coded skeletons with unique color per tracked person
- Bounding boxes with Track ID and activity caption overlay
- Frame counter overlay on output video

---

## Detected Activities

| Activity | Detection Logic |
|---|---|
| **Jumping** | Ankles near hips (airborne) + knees bent |
| **Hands Raised** | Both wrists above shoulders |
| **Hand Raised** | One wrist above shoulder |
| **Sitting** | Knee angle < 115° and hips low |
| **Crouching** | Knee angle < 145° and hips lower than normal standing |
| **Bending** | Hip angle < 140° with straight knees |
| **Running** | Ankle height asymmetry > 18% of body height |
| **Walking** | Ankle height asymmetry 8–18% of body height |
| **Standing** | Default (none of the above) |

---

## Requirements

```
ultralytics
opencv-python
numpy
```

Install with:

```bash
pip install ultralytics opencv-python numpy
```

---

## Setup

1. Place your YOLOv8 Pose model at `Model/yolo26s-pose.pt`
2. Place your input video in the project directory

---

## Configuration

Edit the config section at the top of `pose_video_activity.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `Model/yolo26s-pose.pt` | Path to YOLOv8 Pose model |
| `INPUT_VIDEO` | `group_people.mp4` | Input video file |
| `OUTPUT_VIDEO` | `output_activity.mp4` | Output video file |
| `CONF_THRESH` | `0.3` | Minimum detection confidence |
| `KP_CONF` | `0.3` | Minimum keypoint confidence |
| `SMOOTH_WIN` | `10` | Frames used for activity label smoothing |

---

## Usage

```bash
python pose_video_activity.py
```

The script will print processing progress every 50 frames and save the annotated output video when complete.

---

## Output

The output video contains:
- Colored skeleton drawn on each detected person
- Bounding box per person
- Caption showing `#TrackID Activity` (e.g. `#1 Walking`)
- Frame counter in the top-left corner

---

## Project Structure

```
Pose Detection/
├── pose_video_activity.py   # Main script
├── event_log.json           # Activity event log output
├── Model/
│   └── yolo26s-pose.pt      # YOLOv8 Pose model weights
├── group_people.mp4         # Input video
└── output_activity.mp4      # Output annotated video
```
