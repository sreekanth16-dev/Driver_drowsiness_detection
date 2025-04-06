 🛑 Driver Drowsiness Detection Using CNN and OpenCV

A real-time drowsiness detection system using a Convolutional Neural Network (CNN) model and OpenCV. This project helps prevent road accidents by detecting when a driver’s eyes remain closed for a certain period and raises an audio alert.

## 🎯 Objective

To monitor a driver’s eye status in real-time using webcam footage and raise an alert if drowsiness is detected by analyzing eye closure using a pre-trained deep learning model.

---

## 🧠 How It Works

1. The webcam captures live video frames.
2. Faces and eyes are detected using Haar Cascade classifiers.
3. Each detected eye is passed to a CNN model (`drowsyness_detection1.h5`) to classify as **open** or **closed**.
4. If both eyes remain closed for a threshold number of frames, an alert is triggered using a warning sound.
5. The system can be extended to use services like **Twilio** for making emergency calls (code included but commented).

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – Image processing and webcam interaction
- **TensorFlow / Keras** – For loading the trained CNN model
- **NumPy** – Array manipulation
- **Pygame Mixer** – To play the audio alert
- **Haar Cascades** – For face and eye detection
- **(Optional)** Twilio API – For emergency call notifications (commented out)

---

## 📦 Project Structure
Driver_drowsiness_detection/ ├── drowsiness_detection.py # Main script ├── drowsyness_detection1.h5 # Pre-trained CNN model ├── haarcascade_frontalface_default.xml ├── haarcascade_eye.xml ├── severe-warning-alarm-98704.mp3 ├── README.md



---

## 🚀 Getting Started

### 🔧 Prerequisites

Install the required Python libraries:


```bash
pip install opencv-python numpy tensorflow pygame
If you're using Twilio in the future:
pip install twilio
▶️ Running the Project
Make sure your webcam is connected, and run:

python drowsiness_detection.py









