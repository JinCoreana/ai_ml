import cv2
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.tree import export_graphviz


# Step 2: Extract text from receipt image using OCR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.2/bin/tesseract'
    text = pytesseract.image_to_string(gray_image)
    return text

# Step 5: Train a machine learning model to extract fields (supplier, item name, price, category)


def train_model():
    # Load the labeled dataset
    dataset = pd.read_csv("dataset.csv")
    print(dataset.shape)
    X = dataset[['supplier name', 'item name']]
    y = dataset['category']

    X_encoded = pd.get_dummies(X)
    # Convert text data into numerical features using CountVectorizer
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X_encoded.columns)
    # Train a decision tree model
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_vectorized, y)

    # Train a random forest model
    random_forest = RandomForestClassifier()
    random_forest.fit(X_vectorized, y)

    return decision_tree, random_forest, vectorizer


# Example usage
image_path = "./data/data5.png"

# Step 2: Extract text from receipt image
text = extract_text_from_image(image_path)

# Step 5: Train the machine learning models
decision_tree, random_forest, vectorizer = train_model()

# Convert the extracted text into numerical features using the same vectorizer
text_vectorized = vectorizer.transform([text])

# Predict the category using the decision tree
predicted_category = decision_tree.predict(text_vectorized)[0]

# Predict the category using the random forest
predicted_category_rf = random_forest.predict(text_vectorized)[0]

# Output the predicted category
print("Predicted Category (Decision Tree):", predicted_category)
print("Predicted Category (Random Forest):", predicted_category_rf)

# Visualize the decision tree
dot_data = export_graphviz(decision_tree, out_file=None,
                           feature_names=['supplier name', 'item name'],
                           class_names=decision_tree.classes_,
                           filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)

# Visualize the random forest
for i, estimator in enumerate(random_forest.estimators_):
    dot_data_rf = export_graphviz(estimator, out_file=None,
                                  feature_names=['supplier name', 'item name'],
                                  class_names=estimator.classes_,
                                  filled=True, rounded=True)
    graph_rf = graphviz.Source(dot_data_rf)
    graph_rf.render(f"random_forest_tree_{i}", format="png", cleanup=True)
