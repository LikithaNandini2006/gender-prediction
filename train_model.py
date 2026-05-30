import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

print("="*50)
print("GENDER CLASSIFICATION BY NAME")
print("="*50)

# Create comprehensive dataset of names with genders
data = {
    'name': [
        # Male names
        'John', 'Michael', 'David', 'James', 'Robert', 'William', 'Richard', 'Thomas', 'Charles', 'Daniel',
        'Matthew', 'Anthony', 'Donald', 'Mark', 'Paul', 'Steven', 'Andrew', 'Kenneth', 'Joshua', 'Kevin',
        'Brian', 'George', 'Edward', 'Ronald', 'Timothy', 'Jason', 'Jeffrey', 'Ryan', 'Jacob', 'Gary',
        'Nicholas', 'Eric', 'Jonathan', 'Stephen', 'Larry', 'Justin', 'Scott', 'Brandon', 'Benjamin', 'Samuel',
        'Gregory', 'Frank', 'Alexander', 'Raymond', 'Patrick', 'Jack', 'Dennis', 'Jerry', 'Tyler', 'Aaron',
        'Jose', 'Adam', 'Nathan', 'Henry', 'Douglas', 'Zachary', 'Peter', 'Kyle', 'Walter', 'Ethan',
        'Jeremy', 'Harold', 'Keith', 'Christian', 'Roger', 'Noah', 'Gerald', 'Carl', 'Terry', 'Sean',
        'Austin', 'Arthur', 'Lawrence', 'Jesse', 'Dylan', 'Bryan', 'Joe', 'Jordan', 'Billy', 'Bruce',
        'Albert', 'Willie', 'Gabriel', 'Logan', 'Alan', 'Juan', 'Wayne', 'Roy', 'Ralph', 'Randy',
        'Eugene', 'Vincent', 'Russell', 'Elijah', 'Louis', 'Bobby', 'Philip', 'Johnny', 'Victor',
        
        # Female names
        'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen',
        'Nancy', 'Lisa', 'Betty', 'Margaret', 'Sandra', 'Ashley', 'Kimberly', 'Emily', 'Donna', 'Michelle',
        'Carol', 'Amanda', 'Dorothy', 'Melissa', 'Deborah', 'Stephanie', 'Rebecca', 'Sharon', 'Laura', 'Cynthia',
        'Kathleen', 'Amy', 'Shirley', 'Angela', 'Helen', 'Anna', 'Brenda', 'Pamela', 'Nicole', 'Emma',
        'Samantha', 'Katherine', 'Christine', 'Debra', 'Rachel', 'Carolyn', 'Janet', 'Catherine', 'Maria', 'Heather',
        'Diane', 'Julie', 'Joyce', 'Victoria', 'Kelly', 'Christina', 'Ruth', 'Joan', 'Virginia', 'Judith',
        'Evelyn', 'Hannah', 'Megan', 'Cheryl', 'Andrea', 'Martha', 'Jacqueline', 'Frances', 'Gloria', 'Ann',
        'Teresa', 'Kathryn', 'Sara', 'Janice', 'Jean', 'Alice', 'Madison', 'Doris', 'Abigail', 'Julia',
        'Judy', 'Grace', 'Denise', 'Amber', 'Marilyn', 'Beverly', 'Danielle', 'Theresa', 'Sophia', 'Marie',
        'Diana', 'Brittany', 'Natalie', 'Isabella', 'Charlotte', 'Rose', 'Alexis', 'Kayla', 'Olivia'
    ],
    'gender': [
        # Male labels (first 100 names)
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        
        # Female labels (next 100 names)
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female'
    ]
}

# Keep labels in sync with the names list. The first female entry marks the
# boundary between the male and female names above.
first_female_index = data['name'].index('Mary')
data['gender'] = (
    ['Male'] * first_female_index
    + ['Female'] * (len(data['name']) - first_female_index)
)

# Create DataFrame
df = pd.DataFrame(data)

# Add more name variations (Indian names)
indian_names = [
    ('Raj', 'Male'), ('Rahul', 'Male'), ('Amit', 'Male'), ('Vikram', 'Male'), ('Suresh', 'Male'),
    ('Priya', 'Female'), ('Neha', 'Female'), ('Divya', 'Female'), ('Pooja', 'Female'), ('Anjali', 'Female'),
    ('Aarav', 'Male'), ('Vihaan', 'Male'), ('Vivaan', 'Male'), ('Ananya', 'Female'), ('Ishita', 'Female'),
    ('Aditya', 'Male'), ('Arjun', 'Male'), ('Krishna', 'Male'), ('Lakshmi', 'Female'), ('Saraswati', 'Female'),
    ('Mohammed', 'Male'), ('Ali', 'Male'), ('Hassan', 'Male'), ('Fatima', 'Female'), ('Aisha', 'Female'),
    ('Chen', 'Male'), ('Wei', 'Male'), ('Ming', 'Male'), ('Li', 'Female'), ('Mei', 'Female'),
    ('Carlos', 'Male'), ('Jose', 'Male'), ('Maria', 'Female'), ('Carmen', 'Female'), ('Luisa', 'Female')
]

for name, gender in indian_names:
    df = pd.concat([df, pd.DataFrame({'name': [name], 'gender': [gender]})], ignore_index=True)

print("[OK] Dataset Created Successfully!")
print(f"   Total names: {len(df)}")
print(f"   Male names: {len(df[df['gender']=='Male'])}")
print(f"   Female names: {len(df[df['gender']=='Female'])}")
print("\nSample Data:")
print(df.head(20))

# Feature extraction - convert names to features
def extract_name_features(name):
    """Extract features from name"""
    name = name.lower()
    features = {}
    
    # Length features
    features['name_length'] = len(name)
    
    # First letter
    features['first_letter'] = ord(name[0]) if name else 0
    
    # Last letter
    features['last_letter'] = ord(name[-1]) if name else 0
    
    # Vowel count
    features['vowel_count'] = sum(1 for char in name if char in 'aeiou')
    
    # Consonant count
    features['consonant_count'] = sum(1 for char in name if char.isalpha() and char not in 'aeiou')
    
    # Ends with typical male/female endings
    features['ends_with_a'] = 1 if name.endswith('a') else 0
    features['ends_with_e'] = 1 if name.endswith('e') else 0
    features['ends_with_n'] = 1 if name.endswith('n') else 0
    features['ends_with_y'] = 1 if name.endswith('y') else 0
    
    # Starts with typical letters
    features['starts_with_m'] = 1 if name.startswith('m') else 0
    features['starts_with_j'] = 1 if name.startswith('j') else 0
    features['starts_with_r'] = 1 if name.startswith('r') else 0
    
    return features

# Extract features for all names
feature_list = []
for name in df['name']:
    features = extract_name_features(name)
    feature_list.append(features)

# Create feature DataFrame
X = pd.DataFrame(feature_list)
y = df['gender']

print("\nFeatures extracted:")
print(X.columns.tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print("MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"[OK] Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10))

# Save model and feature columns
joblib.dump(rf_model, BASE_DIR / 'model.pkl')
joblib.dump(X.columns.tolist(), BASE_DIR / 'feature_columns.pkl')

print("\nModel saved as 'model.pkl'")
print("Feature columns saved as 'feature_columns.pkl'")
print(f"\n{'='*50}")
print("[OK] TRAINING COMPLETE!")
print("Next step: Run 'python app.py' to start the web application")
print(f"{'='*50}")

# Test predictions
test_names = ['John', 'Mary', 'Aarav', 'Priya', 'Michael', 'Sarah', 'Raj', 'Neha']
print("\nTesting predictions on sample names:")
for name in test_names:
    features = extract_name_features(name)
    feature_vector = pd.DataFrame([features])[X.columns]
    pred = rf_model.predict(feature_vector)[0]
    prob = rf_model.predict_proba(feature_vector)[0]
    confidence = max(prob) * 100
    print(f"   {name:10} -> {pred:6} (Confidence: {confidence:.1f}%)")
