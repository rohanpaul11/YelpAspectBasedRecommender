import json
from collections import Counter

CLEAR_LINE = '\033[K'

def extract_restaurants(business_file, restaurant_file):
    restaurants = []
    count = 0
    with open(business_file, 'r') as fin:
        for line in fin:
            business = json.loads(line)
            categories = business['categories']
            if categories is None:
                continue
            if 'restaurant'.casefold() in categories.casefold():
                count += 1
                restaurants.append(business)

    print(count, 'restaurants are present.')
    with open(restaurant_file, 'w') as fout:
        for restaurant in restaurants:
            fout.write(json.dumps(restaurant) + '\n')


def extract_reviews_for_restaurants(restaurant_file, review_file, restaurant_review_file):
    restaurant_ids = set()
    with open(restaurant_file, 'r') as fin:
        for line in fin:
            restaurant = json.loads(line)
            restaurant_ids.add(restaurant['business_id'])
    print(len(restaurant_ids), 'restaurant ids.')

    reviews = []
    num_reviews = 0

    max_batch_sz = 10000
    batch_sz = 0
    batch_num = 1

    with open(review_file, 'r') as fin, open(restaurant_review_file, 'w') as fout:
        for line in fin:
            review = json.loads(line)
            if review['business_id'] in restaurant_ids:
                num_reviews += 1
                batch_sz += 1
                reviews.append(review)
            if batch_sz == max_batch_sz:
                print('Writing batch',batch_num,'to file.', end='\r')
                for review in reviews:
                    fout.write(json.dumps(review) + '\n')
                reviews.clear()
                batch_sz = 0
                batch_num += 1
        if batch_sz > 0:
            print('Writing batch',batch_num,'to file.', end='\r')
            for review in reviews:
                fout.write(json.dumps(review) + '\n')
            reviews.clear()
    
    print('{}{} batches'.format(CLEAR_LINE, batch_num))
    print(num_reviews, 'restaurant reviews')


def extract_positive_and_negative_reviews(restaurant_review_file, positive_file, negative_file):
    pos = 0
    neg = 0
    threshold_rating = 3

    pos_reviews = []
    neg_reviews = []

    max_batch_sz = 10000
    batch_sz = 0
    batch_num = 1

    with open(restaurant_review_file, 'r') as fin, open(positive_file, 'w') as fpout, open(negative_file, 'w') as fnout:
        for line in fin:
            review = json.loads(line)            
            if review['stars'] > threshold_rating:
                pos += 1
                pos_reviews.append(review)
            else:
                neg += 1
                neg_reviews.append(review)
            batch_sz += 1
            if batch_sz == max_batch_sz:
                print('Writing batch {} to file, {} positive reviews, {} negative reviews.'.format(batch_num, len(pos_reviews), len(neg_reviews)), end='\r')
                for pos_review in pos_reviews:
                    fpout.write(json.dumps(pos_review) + '\n')
                for neg_review in neg_reviews:
                    fnout.write(json.dumps(neg_review) + '\n')
                batch_sz = 0
                pos_reviews.clear()
                neg_reviews.clear()
                batch_num += 1

        if batch_sz > 0:
            print('Writing batch {} to file, {} positive reviews, {} negative reviews.'.format(batch_num, len(pos_reviews), len(neg_reviews)), end='\r')
            for pos_review in pos_reviews:
                fpout.write(json.dumps(pos_review) + '\n')
            for neg_review in neg_reviews:
                fnout.write(json.dumps(neg_review) + '\n')
            pos_reviews.clear()
            neg_reviews.clear()

    print('{}{} batches'.format(CLEAR_LINE, batch_num))
    print(pos, 'positive restaurant reviews')
    print(neg, 'negative restaurant reviews')


if __name__ == '__main__':
    business_file = '/home/rohan/Documents/yelp_dataset/yelp_academic_dataset_business.json'
    restaurant_file = '../Data/restaurants.json'
    review_file = '/home/rohan/Documents/yelp_dataset/yelp_academic_dataset_review.json'
    restaurant_review_file = '../Data/restaurant_reviews.json'
    positive_review_file = '../Data/positive_reviews.json'
    negative_review_file = '../Data/negative_reviews.json'

    extract_restaurants(business_file, restaurant_file)
    extract_reviews_for_restaurants(
        restaurant_file, review_file, restaurant_review_file)
    extract_positive_and_negative_reviews(
        restaurant_review_file, positive_review_file, negative_review_file)
