import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load your model's predictions
my_submission = pd.read_csv("submission.csv")

# Load ground truth labels
ground_truth = pd.read_csv("gender_submission.csv")

# Merge both on PassengerId
merged = pd.merge(my_submission, ground_truth, on="PassengerId", suffixes=('_pred', '_true'))

# Check merged head
print(merged.head())

# Accuracy
accuracy = accuracy_score(merged['Survived_true'], merged['Survived_pred'])
print("✅ Final Accuracy on Test Set:", accuracy)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(merged['Survived_true'], merged['Survived_pred']))
