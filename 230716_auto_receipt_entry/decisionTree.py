from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the labeled data into a pandas DataFrame
dataset = pd.read_csv('dataset.csv')

# Remove unnecessary columns
dataset = dataset[['supplier', 'item', 'category']]

# Preprocess the text data
dataset['supplier'] = dataset['supplier'].str.lower()
dataset['item'] = dataset['item'].str.lower()

# Convert categorical columns to numerical representation using one-hot encoding
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['supplier'] + ' ' + dataset['item'])

# Split the data into features (X) and target (y)
y = dataset['category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Predict the categories for the test set
y_pred = dt_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print("Decision Trees Results:")
print("Accuracy: {:.2f}".format(accuracy))
print("Classification Report:\n", classification_report)

# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, feature_names=vectorizer.get_feature_names_out().tolist(),
          class_names=dt_classifier.classes_.tolist(), filled=True)
plt.show()
