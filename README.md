# Gender Prediction using Deep Learning

## 📌 Project Overview

Gender Prediction is a Deep Learning and Computer Vision project that predicts whether a detected face belongs to a Male or Female. The system uses a Convolutional Neural Network (CNN) trained on facial image datasets and OpenCV for image processing.

This project demonstrates the practical application of Artificial Intelligence, Deep Learning, and Computer Vision techniques for image classification tasks.

---

## 🚀 Features

- Face Detection using OpenCV
- Gender Classification using CNN
- Image Preprocessing and Normalization
- Model Training and Evaluation
- Real-Time Gender Prediction
- Easy-to-Use Python Implementation
- Deep Learning-Based Classification

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## 📂 Project Structure

```text
gender-prediction/
│
├── dataset/
│   ├── male/
│   └── female/
│
├── model/
│   └── gender_model.h5
│
├── train_model.py
├── recognize.py
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── home.png
    └── result.png
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/LikithaNandini2006/gender-prediction.git
cd gender-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The dataset consists of facial images categorized into:

- Male
- Female

Images are resized, normalized, and preprocessed before being fed into the Convolutional Neural Network.

---

## 🧠 Model Architecture

The model uses a Convolutional Neural Network (CNN) consisting of:

- Convolution Layers
- Max Pooling Layers
- Dropout Layers
- Dense Layers
- Output Layer with Softmax/Sigmoid Activation

The CNN automatically learns facial features that help distinguish between male and female faces.

---

## 🏋️ Training the Model

Run the following command:

```bash
python train_model.py
```

The model will be trained using the dataset and saved as:

```text
gender_model.h5
```

---

## 🔍 Gender Prediction

Run:

```bash
python recognize.py
```

The program will:

1. Load the trained model
2. Read the input image
3. Detect the face
4. Preprocess the image
5. Predict gender
6. Display the result

Example Output:

```text
Prediction: Male
Confidence: 97%
```

or

```text
Prediction: Female
Confidence: 95%
```

---

## 📈 Performance

| Metric | Value |
|----------|----------|
| Accuracy | 95%+ |
| Precision | High |
| Recall | High |
| F1 Score | High |

Performance may vary depending on dataset size and image quality.

---

## 📸 Screenshots

### Prediction Result

```markdown
![Prediction](!<img width="1366" height="768" alt="Screenshot (7)" src="https://github.com/user-attachments/assets/33a5c0aa-b593-43a8-ae3d-f8ca18a5e2c6" />
)

## 💡 Applications

- Smart Attendance Systems
- Human-Computer Interaction
- Demographic Analysis
- Retail Analytics
- Security and Surveillance Systems
- AI-Based Image Processing Solutions

---

## 🎯 Learning Outcomes

Through this project, you can learn:

- Deep Learning Fundamentals
- Computer Vision Basics
- Image Classification
- CNN Architecture Design
- Model Training and Evaluation
- OpenCV Integration
- TensorFlow/Keras Implementation

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push your branch
5. Create a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

