from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Load trained model and feature columns
try:
    model = joblib.load(BASE_DIR / 'model.pkl')
    feature_columns = joblib.load(BASE_DIR / 'feature_columns.pkl')
    print("[OK] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    print("Please run 'python train_model.py' first!")
    exit(1)

def extract_name_features(name):
    """Extract features from name for prediction"""
    name = name.lower().strip()
    features = {}
    
    # Length features
    features['name_length'] = len(name)
    
    # First letter (ASCII value)
    features['first_letter'] = ord(name[0]) if name else 0
    
    # Last letter (ASCII value)
    features['last_letter'] = ord(name[-1]) if name else 0
    
    # Vowel count
    features['vowel_count'] = sum(1 for char in name if char in 'aeiou')
    
    # Consonant count
    features['consonant_count'] = sum(1 for char in name if char.isalpha() and char not in 'aeiou')
    
    # Ends with typical female/male endings
    features['ends_with_a'] = 1 if name.endswith('a') else 0
    features['ends_with_e'] = 1 if name.endswith('e') else 0
    features['ends_with_n'] = 1 if name.endswith('n') else 0
    features['ends_with_y'] = 1 if name.endswith('y') else 0
    
    # Starts with typical letters
    features['starts_with_m'] = 1 if name.startswith('m') else 0
    features['starts_with_j'] = 1 if name.startswith('j') else 0
    features['starts_with_r'] = 1 if name.startswith('r') else 0
    
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get name from form
        name = request.form['name'].strip()
        
        if not name:
            return jsonify({
                'success': False,
                'error': 'Please enter a name'
            })
        
        if not name.replace(' ', '').isalpha():
            return jsonify({
                'success': False,
                'error': 'Please enter a valid name (only letters)'
            })
        
        # Extract features from name
        features = extract_name_features(name)
        
        # Create feature vector with proper column order
        feature_vector = pd.DataFrame([features])[feature_columns]
        
        # Make prediction
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        
        # Get confidence score
        confidence = max(probabilities) * 100
        
        # Get probabilities for each gender
        if model.classes_[0] == 'Male':
            male_prob = probabilities[0] * 100
            female_prob = probabilities[1] * 100
        else:
            male_prob = probabilities[1] * 100
            female_prob = probabilities[0] * 100
        
        # Get similar names for context
        similar_names = get_similar_names(name)
        
        return jsonify({
            'success': True,
            'name': name.title(),
            'gender': prediction,
            'confidence': round(confidence, 1),
            'male_probability': round(male_prob, 1),
            'female_probability': round(female_prob, 1),
            'similar_names': similar_names
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        })

def get_similar_names(name):
    """Get similar names for context"""
    name_lower = name.lower()
    common_names = {
        'john': ['John', 'Johnny', 'Jonathan'],
        'mary': ['Mary', 'Maria', 'Marilyn'],
        'michael': ['Michael', 'Mike', 'Micheal'],
        'jennifer': ['Jennifer', 'Jenny', 'Jenna'],
        'david': ['David', 'Dave', 'Davi'],
        'sarah': ['Sarah', 'Sara', 'Sariah'],
        'raj': ['Raj', 'Rajan', 'Rajesh'],
        'priya': ['Priya', 'Priyanka', 'Priyanshi']
    }
    
    if name_lower in common_names:
        return common_names[name_lower]
    elif name_lower.endswith('a'):
        return [name.title(), name.title() + ' (Female pattern)']
    elif name_lower.endswith('n') or name_lower.endswith('d'):
        return [name.title(), name.title() + ' (Male pattern)']
    else:
        return [name.title()]

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
