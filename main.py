# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv("data.csv")

print("First 5 Rows:\n")
print(data.head())

# -----------------------------
# Check Missing Values
# -----------------------------

print("\nMissing Values:\n")
print(data.isnull().sum())

# -----------------------------
# Convert Result Column
# -----------------------------

data['Result'] = data['Result'].map({'Fail': 0, 'Pass': 1})

# -----------------------------
# Data Visualization
# -----------------------------

sns.countplot(x='Result', data=data)
plt.title("Pass vs Fail")
plt.show()

# Heatmap
sns.heatmap(data.corr(), annot=True)
plt.show()

# -----------------------------
# Features and Target
# -----------------------------

X = data[['Hours', 'Attendance', 'PreviousMarks']]
y = data['Result']

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model Training
# -----------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# -----------------------------
# User Input Prediction
# -----------------------------

print("\nEnter Student Details")

hours = float(input("Study Hours: "))
attendance = float(input("Attendance: "))
marks = float(input("Previous Marks: "))

new_data = pd.DataFrame({
    'Hours': [hours],
    'Attendance': [attendance],
    'PreviousMarks': [marks]
})

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")