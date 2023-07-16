import csv
import random

categories = ['sales', 'other income', 'purchase',
              'direct expense', 'overheads', 'other expenses', 'taxation']
types = ['+', '+', '-', '-', '-', '-', '-']

# Supplier names and item categories
supplier_names = ['Acme Corporation', 'Global Enterprises', 'Apex Solutions', 'Pioneer Industries', 'Infinite Ventures', 'Elevate Inc', 'Strategic Holdings',
                  'ABC Traders', 'XYZ Company', 'Sunrise Enterprises', 'Dynamic Systems', 'Mega Corp', 'Alpha Ltd', 'Omega Solutions',
                  'Tech Innovators', 'Quality Supplies', 'Prime Industries', 'Powerful Systems', 'Bravo Corporation', 'Sunset Enterprises']

item_categories = {
    'sales': ['Electronics', 'Software Solutions', 'Consulting Services', 'Industrial Equipment', 'Commercial Vehicles'],
    'other income': ['Training Programs', 'Intellectual Property', 'Event Management', 'Licensing Fees'],
    'purchase': ['Raw Materials', 'Office Equipment', 'Machinery', 'Furniture', 'Computers'],
    'direct expense': ['Labor Costs', 'Shipping Charges', 'Utilities', 'Maintenance Services'],
    'overheads': ['Rent', 'Insurance', 'Taxes', 'Advertising'],
    'other expenses': ['Travel Expenses', 'Marketing Campaigns', 'Professional Fees'],
    'taxation': ['Income Tax', 'Sales Tax', 'Property Tax']
}

# Generate random cases for the dataset with diverse supplier names
dataset = []
for i in range(100):
    date = f"{random.randint(2023, 2024)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
    category = random.choice(categories)
    # Randomly select from the supplier names pool
    supplier_name = random.choice(
        supplier_names)
    item_name = random.choice(item_categories[category])
    price = round(random.uniform(30, 800), 2)
    type = types[categories.index(category)]

    case_number = f"{i+1}"
    entry = [case_number, date, item_name,
             price, supplier_name, category, type]
    dataset.append(entry)

# Write the dataset to a CSV file
with open('dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['number', 'date', 'item name', 'price',
                    'supplier name', 'category', 'type'])  # Write header row
    writer.writerows(dataset)
