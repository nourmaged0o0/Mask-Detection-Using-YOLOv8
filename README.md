# Mask Detection Using YOLOv8

This project implements a real-time mask detection system that identifies individuals wearing or not wearing masks. If a person without a mask is detected, their picture is captured and saved in a designated folder. This solution is ideal for use in environments like hospitals, pharmacies, or public spaces to ensure compliance with mask-wearing policies.

## Project Overview

- **Model**: YOLOv8
- **Task**: Detect mask-wearing individuals, save images of those without masks.
- **Application**: Useful for monitoring mask-wearing in public places.

## Installation

### Clone the Repository
To start using the project, clone this repository:

```bash
git clone https://github.com/yourusername/mask-detection-yolov8.git
cd mask-detection-yolov8
```

### Set up Virtual Environment

1. Create and activate a virtual environment:
   ```bash
   python -m venv yolo_env
   source yolo_env/bin/activate  # Windows: yolo_env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Download the Model
Ensure the best YOLO model (`best.pt`) is saved in the correct path. You can find this in the repository under `runs/detect/train14/weights/best.pt`.

## Dataset

The dataset used in this project includes mask-wearing and non-mask-wearing images. You can modify the dataset location in the `data.yaml` file if you wish to retrain the model with new data or if you want to use the base dataset here is the link: (https://universe.roboflow.com/joseph-nelson/mask-wearing/dataset/18).

## Usage

### Running Mask Detection on Video
Use the provided script to run the mask detection model on a video feed:

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")  # Load pre-trained model
cap = cv2.VideoCapture("path_to_video.mp4")

# Process the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, conf=0.6)
    # Additional code for saving images of non-mask wearers...
```

### Saving Images of People Without Masks
The system captures and saves pictures of detected individuals without masks into a specific folder. These images are labeled with unique IDs.

### Configuration
- You can adjust the confidence threshold, detection speed, and model path in the provided Python script.

## Results and Evaluation

Several metrics were used to evaluate the model’s performance:

- **F1 Score Curve**: ![F1_curve](https://github.com/user-attachments/assets/915bfdd2-9b42-4439-850a-8badbdcc7616)
- **Precision-Recall Curve**: ![PR_curve](https://github.com/user-attachments/assets/e2c20bec-be75-433d-b3c3-e9d1343eedbc)
- **Confusion Matrix**: ![confusion_matrix](https://github.com/user-attachments/assets/806011ce-a9dd-400d-a393-998025ed3a8c)

These metrics demonstrate the model's ability to balance precision and recall when identifying individuals with and without masks.

## Example of saved images:


  ![240](https://github.com/user-attachments/assets/94a04f25-919a-4052-9eed-1af969bde15a) ![1](https://github.com/user-attachments/assets/ba71823a-da64-4b68-b79d-d09acc6ec677) ![128](https://github.com/user-attachments/assets/d0082453-8869-4c05-814c-a5014e2d5f94)

## Training

To retrain the model on your own dataset:

1. Modify the `data.yaml` file to point to your dataset.
2. Use the following command to start training:

```python
model.train(
    data="path_to_data.yaml", 
    epochs=60, 
    imgsz=640  # Image size
)
```

Training results will be saved under the `runs/detect/train15` directory.

## Sample Output

Sample output from the model detecting individuals in a video stream is shown below:
- ![Annotated Video](https://github.com/user-attachments/assets/7e4fc170-2b02-4d9f-8b29-624231b7480f)
