Certainly! Below is an updated version of your script along with a section for requirements installation and a more comprehensive GitHub README file.

### `README.md`

---

# Football Player Detection with YOLOv11

This project uses a fine-tuned YOLO (You Only Look Once) model to detect football players in videos. It leverages the `ultralytics` YOLOv5 model to perform object detection in video files, and the model is trained on a custom dataset to recognize football players.

## Installation

To get started with this project, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/football-player-detection.git
cd football-player-detection
```

### 2. Install dependencies

Make sure you have Python 3.7+ installed. Create and activate a virtual environment if preferred.

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt
```

### 3. Requirements

You can also manually install the required libraries if you don't want to use `requirements.txt`:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install ultralytics
```

### 4. Additional Libraries (for training):

If you want to train or fine-tune the model, ensure these libraries are installed:

```bash
pip install matplotlib seaborn
```

## Usage

### 1. Load the pre-trained model and detect football players in videos

```python
from ultralytics import YOLO
import cv2 as cv

# Define the path to your YOLO model
model_path = "../models/yolo11x.pt"
model = YOLO(model_path)  # Load the pre-trained YOLO model

# Move model to GPU (if available)
model.to('cuda')

# Run prediction on the input video
results = model.predict('../Resource/Videos/08fd33_4.mp4', save=True)

# Print the results
print(results[0])

# Print detected bounding boxes
for box in results[0].boxes:
    print(box)
```

### 2. Fine-tuning the YOLO model

You can fine-tune the model on a custom dataset of football players using the following command.

```bash
!yolo task=detect mode=train model="../models/yolo11x.pt" data='../Datasets/football-players-detection.v12i.yolov11/data.yaml' epochs=100 imgsz=640
```

- `data.yaml` is a configuration file that contains the paths to the training dataset and class names.
- `epochs=100` defines the number of training epochs.
- `imgsz=640` defines the input image size for training.

Make sure you replace the paths with the correct ones for your setup.

### 3. Testing on a New Video

To run detection on a new video, simply change the path to your video file and run the prediction again:

```python
results = model.predict('path_to_your_video.mp4', save=True)
```

This will save the output video with detections.

## Model Training

To fine-tune the model on your custom dataset (e.g., football players), you can use the following command to train the model. This assumes that you have already prepared a dataset in YOLOv5 format.

```bash
yolo task=detect mode=train model="../models/yolo11x.pt" data='../Datasets/football-players-detection.v12i.yolov11/data.yaml' epochs=100 imgsz=640
```

For more information on how to prepare the dataset, check the [YOLOv5 documentation](https://github.com/ultralytics/yolov5).

## Example Output

After running the video detection, the output will look like this:

```bash
Results:
   - Detected football player(s) in frames.
   - Bounding box coordinates, confidence score, and class labels will be printed.
```

For each detected object, you will get its bounding box coordinates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### `requirements.txt`

```txt
torch>=1.9.0
opencv-python
ultralytics
matplotlib
seaborn
```

---

This README file provides an overview of the project, installation instructions, how to run the detection script, fine-tune the model, and example outputs. Let me know if you'd like further adjustments!
