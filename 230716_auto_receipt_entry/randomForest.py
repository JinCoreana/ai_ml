import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD

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

# Train the Random Forest model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Predict the categories for the test set
y_pred = rf_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print("Random Forest Results:")
print("Accuracy: {:.2f}".format(accuracy))
print("Classification Report:\n", classification_report)

# Create a color mapping for the categories
category_mapping = {category: i for i,
                    category in enumerate(dataset['category'].unique())}
color_mapping = [category_mapping[category] for category in y_test]

# Plot the decision boundary for Random Forest using TruncatedSVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_test.toarray())

# Create a color mapping for the categories
category_mapping = {category: i for i,
                    category in enumerate(dataset['category'].unique())}

# Plot the decision boundary for Random Forest
h = .02  # Step size in the mesh
x_min, x_max = X_svd[:, 0].min() - 1, X_svd[:, 0].max() + 1
y_min, y_max = X_svd[:, 1].min() - 1, X_svd[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = rf_classifier.predict(svd.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))

# Convert categorical labels in Z to numerical representations using the mapping
Z = np.array([category_mapping[label] for label in Z])

# Reshape Z to match xx and yy shapes
Z = Z.reshape(xx.shape)
# Plot the decision boundary for Random Forest
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

for category, color in category_mapping.items():
    category_mask = y_test == category
    plt.scatter(X_svd[category_mask, 0],
                X_svd[category_mask, 1], label=category)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Random Forest Decision Boundary')
plt.legend()
plt.show()
