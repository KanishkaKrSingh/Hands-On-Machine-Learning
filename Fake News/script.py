import numpy as np  # numpy array
import pandas as pd  # dataframes
import re   # searching words
from nltk.corpus import stopwords  # words that have no value
from nltk.stem.porter import PorterStemmer   # Stemming words - root word fo
from sklearn.feature_extraction.text import TfidfVectorizer  # convert text into Feature Vectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
# Printing the stop words that are useless
print(stopwords.words('english'))

# ------------- Data Pre-processing ------------------

# loading dataset to panda dataframe

news_dataset = pd.read_csv('train.csv')
news_dataset.head()

# counting the number of missing values in dataset
news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')

# Merging author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

# separating data and label
X = news_dataset.drop(['label'], axis=1)
Y = news_dataset['label']

# Stemming Word --> Root Word
# Function work --> sub() is searching for given Regex within the text we are given, then
# it lowers it and splits into words, and then we stem them one by one skipping STOP WORDS
porter = PorterStemmer()


def stem(x):
    stemmed_x = re.sub('[^a-zA-Z]', ' ', x)
    stemmed_x = stemmed_x.lower()
    stemmed_x = stemmed_x.split()
    stemmed_x = [porter.stem(word) for word in stemmed_x if not word in stopwords.words('english')]
    return ' '.join(stemmed_x)


news_dataset['content'] = news_dataset['content'].apply(stem)

# Separating the data dnd label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Converting textual data to numerical data --> Vectorizer
vectorizer = TfidfVectorizer()  # Term Freq based vector that gives priority to word that occurs often
vectorizer.fit(X)
X = vectorizer.transform(X)

# Splitting dataset into
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Training the data using Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy scoring the dat to see our model's efficiency
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Training Accuracy:', training_accuracy)

X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Testing Accuracy:', testing_accuracy)

# Confusion Matrix
cm = confusion_matrix(Y_test, X_test_prediction)

# Plotting the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(Y_test, X_test_prediction, target_names=['Real', 'Fake']))

# Making a Predictive System

X_new = X_test[0]
print(f"The news was --> \n {X_test[0]} \n and the Expected label is --> \n {Y_test[0]}")
prediction = model.predict(X_new)
print(prediction)

if prediction == 0:
    print("News is REAL")
else:
    print("News is Fake")