import csv
import random
import string

categories = ['sales', 'other income', 'purchase',
              'direct expense', 'overheads', 'other expenses', 'taxation']
types = ['+', '+', '-', '-', '-', '-', '-']

# Generate a random string of given length


def random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


# Generate random cases for the dataset with diverse supplier names
dataset = []
for i in range(100):
    date = f"{random.randint(2023, 2024)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
    category = random.choice(categories)

    # Generate unique supplier name by combining random string with unique identifier
    supplier_name = f"Supplier-{i+1}-{random_string(4)}"

    # Generate unique item name by combining random string with unique identifier
    item_name = f"Item-{i+1}-{random_string(4)}"

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
