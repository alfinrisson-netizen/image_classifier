# Image Classifier using CNN + Streamlit

A deep learning project that classifies images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed using Streamlit.

---

## Project Overview

This project is an **image classification web app** that can predict the category of an image using a trained CNN model.
It is trained on the **CIFAR-10 dataset**, which contains 10 different classes.

The web app allows users to upload an image and instantly get predictions with confidence scores.

---

##  Dataset

The model is trained on the CIFAR-10 dataset which includes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

Dataset is automatically downloaded via TensorFlow.

---

##  Model Architecture

The CNN model consists of:

* Convolutional layers (Conv2D)
* MaxPooling layers
* Flatten layer
* Dense (Fully Connected) layers
* Softmax output layer

Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metric: Accuracy

---

##  Technologies Used

* Python 
* TensorFlow / Keras
* NumPy
* Matplotlib
* Streamlit
* PIL (Python Imaging Library)

---

##  Project Structure

```
image_classifier/
│
├── app.py                  # Streamlit web app
├── image_classifier.keras # Trained model
├── train.py               # Model training script
├── requirements.txt       # Dependencies
├── .gitignore             # Ignored files
└── README.md              # Project documentation
```

---

##  Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/image_classifier.git
cd image_classifier
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Run the Project

### Train the model (optional)

```bash
python train.py
```

### Run Streamlit app

```bash
streamlit run app.py
```

---

##  Features

* Upload images from device
* Real-time image classification
* Confidence score display
* Simple and interactive UI
* Lightweight CNN model

---

##  Results

* Achieved ~70% accuracy on CIFAR-10 dataset
* Model performs well on general image categories
* Real-time predictions using Streamlit UI

---

##  Future Improvements

* Use Transfer Learning (MobileNet / ResNet)
* Improve accuracy to 85–95%
* Add drag & drop image upload
* Deploy on Streamlit Cloud
* Add Grad-CAM visualization

---

##  Author

**Alfin Risson**

* GitHub: https://github.com/alfinrisson-netizen
* Project Type: Beginner Deep Learning + Web App

---

This is a beginner-friendly deep learning project built for learning CNNs, image classification, and deployment using Streamlit.
