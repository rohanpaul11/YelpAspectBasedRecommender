import json

def filter():
    with open('../Data/no_aspect_reviews.txt', 'r') as fin:
        review_ids = set()
        for line in fin:
            review_ids.add(line.strip())
    
    with open('../Data/filtered_dataset.json', 'r') as fin, open('../Data/buggy_reviews.txt', 'w') as fout:
        for line in fin:
            review = json.loads(line)
            if review['review_id'] in review_ids:
                fout.write(json.dumps(review) + '\n')
    
    with open('../Data/dependencies.txt', 'r') as fin, open('../Data/buggy_dependencies.txt', 'w') as fout:
        for line in fin:
            review = line.split(':')[0].split()[1]
            if review in review_ids:
                fout.write(json.dumps(line) + '\n')

if __name__ == '__main__':
    filter()