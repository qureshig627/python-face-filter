Python Real-Time Face Filter 🕶️
A Python application that applies a stabilized AR filter to faces in real-time from a video file or webcam feed. This project uses OpenCV for computer vision tasks and dlib for accurate facial landmark detection.

Gif created from output video : ![Demo GIF of the face filter in action](output_demo.gif)

✨ Features
Real-Time Face Detection: Identifies one or more faces in every frame of a video stream.

Facial Landmark Detection: Pinpoints 68 specific points on the face (eyes, nose, mouth) for precise filter placement.

Filter Rotation: Automatically calculates the angle of the head and rotates the filter to match, ensuring it always looks natural.

Jitter Smoothing: Implements a temporal smoothing filter (moving average) on the landmark data to eliminate jitter and create a stable, "locked-on" effect.

Dynamic Resizing: Scales the filter appropriately based on the detected size of the face.

Customizable Filters: Easily use any transparent PNG image as a filter.

Timestamp Overlay: Adds a timestamp to the output video.

🛠️ Setup & Installation
Follow these steps to get the project running on your local machine.

1. Clone the Repository

First, clone this repository to your local machine.

git clone [https://github.com/your-username/python-face-filter.git](https://github.com/your-username/python-face-filter.git)
cd python-face-filter

2. Install Dependencies

This project requires a few Python libraries. You can install them using pip.

pip install opencv-python numpy dlib tkinter

3. Download the dlib Landmark Model

The facial landmark detector requires a pre-trained model file. This file is not included in the repository due to its large size.

Download the file here: shape_predictor_68_face_landmarks.dat.bz2

Unzip the file. After unzipping, you will have a file named shape_predictor_68_face_landmarks.dat.

Place the .dat file in the root directory of this project (the same folder as the Python script).

🚀 How to Run
Once the setup is complete, you can run the script from your terminal.

python Python_Snapchat_Vision_Filter.py

The script will prompt you with the following choices:

Choose input source: Select 1 for your webcam or 2 to process a video file.

Select a video file (if you chose 2): A file dialog will open for you to choose a video.

Select a filter image: A file dialog will open for you to choose a transparent .png file to use as the filter.

The processed video will be saved as output_video.mp4 or output_webcam.mp4 in the project directory.

🔬 How It Works
Face Detection: For each frame of the video, dlib.get_frontal_face_detector() is used to locate faces.

Landmark Prediction: The shape_predictor model is then used on the face region to identify the 68 facial landmarks.

Smoothing: To prevent jitter, a Stabilizer class calculates a moving average of the eye positions, width, and angle over the last few frames. This provides a "stable" target instead of the raw, noisy measurements from the detector.

Transformation Calculation:

The angle is calculated based on the vector between the stable eye positions.

The size of the filter is calculated based on the stable distance between the eyes.

The position is calculated based on the stable center point of the eyes.

Filter Application: The filter image is resized and rotated according to the stabilized calculations. It is then overlaid onto the video frame using an alpha mask to handle transparency correctly.

🙏 Acknowledgements
This project was developed with assistance from Google's Gemini.

Special thanks to the developers of OpenCV and dlib for their powerful open-source libraries.
