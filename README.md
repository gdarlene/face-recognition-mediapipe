## AI not ML Face Recognition using MediaPipe Framework
This project shows the usefulness of using mediapipe framework for face recognition with Artificial intelligence that doesn't need deep leaning ro recognise various face features
## Installation

```bash
# Create virtual environment
python -m venv .venv

# Windows
./.venv/Scripts/activate 

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
* Run the following python scripts to create the dataset with only 50 images but you can modify to capture more as you wish. 
```bash
python createdataset.py
python 01_train.py
python 02_recognize.py
python 03_track_face.py
python 04_track_face.py
python 05_track_face.py
```

## Explanation

### 01_create_dataset.py
Take images of faces and train the model as datasets. Make sure you are not in motion while taking images and there is no light changes. Too much light or dark changes can cause the model to be unstable. Take as many pictures as you can (Minimum 10 images).

### 02_review_dataset.py
Review the dataset and remove any images that are not clear.

### 03_train_model.py
Train the model using the dataset.

### 04_predict.py
Predict the face using the trained model with the haarcascade algorithm. 

### 05_track_face.py
Track the face motion and direction