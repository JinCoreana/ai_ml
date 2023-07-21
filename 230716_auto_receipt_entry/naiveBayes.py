import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA

# Load the labeled data into a pandas DataFrame
dataset = pd.read_csv('dataset.csv')

# Remove unnecessary columns
dataset = dataset[['supplier', 'item', 'category']]

# Preprocess the text data
dataset['supplier'] = dataset['supplier'].str.lower()
dataset['item'] = dataset['item'].str.lower()

# Combine supplier and item columns for text representation
dataset['text'] = dataset['supplier'] + ' ' + dataset['item']

# Convert categorical columns to numerical representation using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['text'])

# Take the absolute value of the elements in the sparse matrix
X = X.astype(np.float64)  # Ensure X is of float type
X.data = np.abs(X.data)   # Take the absolute value of non-zero elements

# Split the data into features (X) and target (y)
y = dataset['category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Reduce data to 2D using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train.toarray())
X_test_pca = pca.transform(X_test.toarray())

# Train the Naive Bayes model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# ... (calculate evaluation metrics and print results, if needed)

# Plot the decision boundary
category_mapping = {category: i for i,
                    category in enumerate(dataset['category'].unique())}

# Convert X_test and y_test to NumPy arrays if needed
X_test_pca = np.array(X_test_pca)
y_test = np.array(y_test)

plt.figure(figsize=(10, 6))

# Rescale the PCA components to ensure all data points are within the visible plot area
x_scaler = (X_test_pca[:, 0].max() - X_test_pca[:, 0].min()) / 10
y_scaler = (X_test_pca[:, 1].max() - X_test_pca[:, 1].min()) / 10
x_min, x_max = X_test_pca[:, 0].min(
) - x_scaler, X_test_pca[:, 0].max() + x_scaler
y_min, y_max = X_test_pca[:, 1].min(
) - y_scaler, X_test_pca[:, 1].max() + y_scaler

# Generate a refined meshgrid for the decision boundary
h = .01  # Step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = nb_classifier.predict(vectorizer.transform(
    np.c_[xx.ravel(), yy.ravel()].astype(np.float64)).abs())

# Convert categorical labels in Z to numerical representations using the mapping
Z = np.array([category_mapping[label] for label in Z])

# Reshape Z to match xx and yy shapes
Z = Z.reshape(xx.shape)

# Plot the decision regions with increased opacity
plt.contourf(xx, yy, Z, alpha=1.0, cmap='viridis')

# Plot the scatter plot dots for each category on top of the decision boundary
colors = ['red', 'blue', 'green', 'purple',
          'orange']  # Add more colors if needed
for category, i in category_mapping.items():
    plt.scatter(X_test_pca[y_test == i, 0],
                X_test_pca[y_test == i, 1], label=category,
                marker='o', color=colors[i % len(colors)], edgecolors='black', linewidth=1)

# Set the limits for the plot area
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Naive Bayes Decision Boundary (PCA)')
plt.legend()
plt.show()
