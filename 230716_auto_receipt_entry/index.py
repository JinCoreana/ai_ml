import cv2
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import re

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

    # Convert text data into numerical features using CountVectorizer
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(dataset["item"])

    # Encode the categorical "category" variable

    model_category = LogisticRegression()
    model_category.fit(X_vectorized, dataset["category"])

    model_type = LogisticRegression()
    model_type.fit(X_vectorized, dataset["type"])

    return (model_category, model_type), vectorizer


# Example usage
image_path = "./data/data6.png"

# Step 2: Extract text from receipt image
text = extract_text_from_image(image_path)

# Step 5: Train the machine learning models
models, vectorizer = train_model()

# Convert the extracted text into numerical features using the same vectorizer
text_vectorized = vectorizer.transform([text])

# Predict the fields (supplier name, item name, price, category) using the trained models

predicted_category = models[0].predict(text_vectorized)[0]
predicted_type = models[1].predict(text_vectorized)[0]

supplier_name = ""
supplier_pattern = r"([A-Z]+\s?,\s?[A-Z]+\s?[A-Z]+)"
supplier_match = re.search(supplier_pattern, text, re.IGNORECASE)
if supplier_match:
    supplier_name = supplier_match.group(1)

date = ""
date_pattern = r"Date/Time\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})"
date_match = re.search(date_pattern, text, re.IGNORECASE)
if date_match:
    date = date_match.group(1)

price_values = []
price_pattern = r"TOTAL CHARGES\s+([\d.]+)"
price_match = re.search(price_pattern, text, re.IGNORECASE)
if price_match:
    price_values.append(float(price_match.group(1)))

item_names = []
item_pattern = r"(Vehicle Hy|Odometer Out\.\s?\d+|Odometer In:\s?\d+|Fuel Reading)"
item_matches = re.findall(item_pattern, text)
if item_matches:
    item_names = item_matches


# Output the predicted fields
print("Predicted Supplier Name:", supplier_name, text)
print("Predicted Item Name:", item_names)
print("Predicted Price:", price_values)
print("Predicted Category:", predicted_category)
print("Predicted Date:", date)
print("Predicted Type:", predicted_type)
