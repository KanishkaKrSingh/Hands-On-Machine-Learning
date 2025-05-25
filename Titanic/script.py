import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ========================== TRAINING ============================

# Load dataset
df = pd.read_csv("train.csv")

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Encode categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Save PassengerId separately (optional)
passenger_ids_train = df['PassengerId']

# Define X and y
X = df.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
y = df['Survived']

# Save column order for test data alignment
X_train_columns = X.columns

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\nLogistic Regression Accuracy:", model.score(X_test, y_test))
print(classification_report(y_test, model.predict(X_test)))

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
print("\nDecision Tree Accuracy:", tree_model.score(X_test, y_test))
print(classification_report(y_test, tree_model.predict(X_test)))

# Random Forest
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
print("\nRandom Forest Accuracy:", forest_model.score(X_test, y_test))
print(classification_report(y_test, forest_model.predict(X_test)))

# Grid Search for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("\nGridSearchCV Best Parameters:", grid_search.best_params_)
print("GridSearchCV Best CV Score:", grid_search.best_score_)

# ======================= VISUALIZATION =========================

sns.countplot(x='Sex_male', hue='Survived', data=df)
plt.title("Survival Count by Gender")
plt.show()

sns.histplot(data=df, x='Age', hue='Survived', multiple='stack')
plt.title("Age Distribution by Survival")
plt.show()

sns.boxplot(x='Sex_male', y='Age', hue='Survived', data=df)
plt.title("Boxplot: Age & Gender vs Survival")
plt.show()

# ======================== TESTING ==============================

# Load test data
test_data = pd.read_csv("test.csv")

# Fill missing values
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

# Encode categorical features like train data
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Fix any missing columns (alignment with training features)
for col in X_train_columns:
    if col not in test_data.columns:
        test_data[col] = 0

# Ensure correct column order
test_data = test_data[X_train_columns]

# Save PassengerId
passenger_ids = test_data['PassengerId'] if 'PassengerId' in test_data.columns else pd.read_csv("test.csv")['PassengerId']

# Predict
predictions = forest_model.predict(test_data)

# Save submission
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})
submission.to_csv("submission.csv", index=False)
print("\n✅ Submission file created as 'submission.csv'")
