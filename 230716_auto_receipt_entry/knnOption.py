import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

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
pca = PCA(n_components=2)
X = pca.fit_transform(X.toarray())

# Split the data into features (X) and target (y)
y = dataset['category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Train the KNN classifier
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict the categories for the test set
y_pred = knn.predict(X_test)

# Print the KNN graph
print("KNN Graph:")
print(knn.kneighbors_graph(X_test))

# Calculate evaluation metrics
jaccard_index = metrics.jaccard_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Jaccard Index: {:.2f}".format(jaccard_index))
print("F1 Score: {:.2f}".format(f1_score))

# Evaluate the model
classification_report = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report)

# Get unique categories
categories = dataset['category'].unique()


# Create a color mapping for the categories
category_mapping = {category: i for i,
                    category in enumerate(dataset['category'].unique())}
color_mapping = [category_mapping[category] for category in y_test]

# Plot the decision boundary
plt.figure(figsize=(10, 6))
for category, color in category_mapping.items():
    plt.scatter(X_test[y_test == category, 0], X_test[y_test == category, 1],
                label=category)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Decision Boundary')
plt.legend()
plt.show()
