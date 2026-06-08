# gender-predictionGender Prediction using Deep Learning

A Deep Learning-based Gender Prediction System that detects faces from images and predicts whether the person is Male or Female using a Convolutional Neural Network (CNN).

📌 Project Overview

This project uses Computer Vision and Deep Learning techniques to:

Detect faces from input images
Preprocess facial images
Predict gender using a trained CNN model
Display prediction results with confidence scores

The project is built using Python, TensorFlow/Keras, OpenCV, and NumPy.

🚀 Features
Face Detection using OpenCV
Gender Classification using CNN
Real-time Image Prediction
Easy-to-use Python Implementation
Deep Learning Model Training Support
🛠️ Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Scikit-learn
📂 Project Structure
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
└── screenshots/
⚙️ Installation
1. Clone the Repository
git clone https://github.com/LikithaNandini2006/gender-prediction.git
cd gender-prediction
2. Create Virtual Environment
python -m venv venv
3. Activate Virtual Environment

Windows:

venv\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
▶️ Train the Model
python train_model.py

The trained model will be saved as:

gender_model.h5
🖼️ Run Gender Prediction
python recognize.py

Upload or provide an image, and the model will predict:

Prediction: Male
Confidence: 97%

or

Prediction: Female
Confidence: 95%
📊 Dataset

The dataset contains facial images categorized into:

Male
Female

Images are preprocessed and resized before training.

📈 Model Performance
Metric	Score
Accuracy	95%+
Loss	Low
Precision	High
Recall	High

Actual results may vary depending on dataset quality and size.

 Screenshots


### Home Screen

![Home](<img width="1366" height="768" alt="Screenshot (7)" src="https://github.com/user-attachments/assets/df780e07-5858-49a7-af60-485cf4807585" />
)

### Prediction Result

![Prediction](screenshots/result.png)

🤝 Contributing

Contributions are welcome.

Fork the repository
Create a new branch
Commit your changes
Push the branch
Open a Pull Request
📜 License

This project is open-source and available under the MIT License.
