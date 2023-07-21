import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

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

# Train the SVM model
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

# Predict the categories for the test set
y_pred = svm_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print("SVM Results:")
print("Accuracy: {:.2f}".format(accuracy))
print("Classification Report:\n", classification_report)

# Plot the decision boundary
category_mapping = {category: i for i,
                    category in enumerate(dataset['category'].unique())}
plt.figure(figsize=(10, 6))
for category, color in category_mapping.items():
    plt.scatter(X_test[y_test == category, 0], X_test[y_test == category, 1],
                label=category)

# Plot the decision boundary for SVM
h = .02  # Step size in the mesh
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.jet)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.legend()
plt.show()
