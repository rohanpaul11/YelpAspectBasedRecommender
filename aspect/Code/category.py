import json
import ast
from collections import defaultdict

def categories(restaurant_file, category_file):
    all_categories = defaultdict(list)
    with open(restaurant_file, 'r') as fin, open(category_file, 'w') as fout:
        for line in fin:
            restaurant = json.loads(line)
            id = restaurant['business_id']
            categories = restaurant['categories'].split(',')
            for category in categories:
                all_categories[category.strip().lower()].append(id)
        for category in sorted(all_categories.keys()):
            fout.write(category + ':' + str(all_categories[category]) + '\n')

if __name__ == '__main__':
    restaurant_file = '../Data/restaurants.json'
    category_file = '../Data/categories.txt'
    categories(restaurant_file, category_file)
