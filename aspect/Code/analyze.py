import json
from collections import defaultdict, Counter
import server
from nltk.parse import CoreNLPParser
import sys
from matplotlib import pyplot as plt
import random
import ast

CLEAR_LINE = '\033[K'

def analyze_business(business_file, stop_category_file, business_stat_file):
    num_businesses = 0
    num_businesses_without_stop = 0
    num_businesses_with_stop = 0

    city_business_map = defaultdict(int)
    city_restaurant_map = defaultdict(int)
    category_map = defaultdict(int)    
    business_city_map = defaultdict(str)

    cat_for_non_restaurant_businesses = defaultdict(int)
    cat_for_restaurant_businesses = defaultdict(int)

    businesses_not_restaurants = []
    restaurants = []

    stop_categories = []
    with open(stop_category_file, 'r') as fin:
        for line in fin:
            stop_categories.append(line.strip().lower())
            
    with open(business_file, 'r') as fin:
        for line in fin:
            business = json.loads(line)
            
            num_businesses += 1

            city = business['city'].lower()
            city_business_map[city] += 1
            business_id = business['business_id']

            business_city_map[business_id] = city

            categories = business['categories']
            if categories is None:
                continue
            
            categories = categories.lower()
            all_cat = categories.strip().split(',')
            for cat in all_cat:
                category_map[cat.strip()] += 1

            if 'restaurant'.casefold() in categories:
                num_businesses_without_stop += 1

                ok = True
                for stop_cat in stop_categories:                    
                    if stop_cat in categories:
                        ok = False
                        break

                if ok:
                    num_businesses_with_stop += 1
                    restaurants.append((business_id, business['name']))
                    city_restaurant_map[city] += 1
                    for cat in all_cat:
                        cat_for_restaurant_businesses[cat.strip()] += 1
                else:                    
                    businesses_not_restaurants.append((business_id, business['name']))
                    for cat in all_cat:
                        cat_for_non_restaurant_businesses[cat.strip()] += 1
                
    
    with open(business_stat_file, 'w') as fout:
        fout.write('Total number of businesses = {}\n'.format(num_businesses))        
        fout.write('Total number of cities = {}\n'.format(len(city_business_map)))
        fout.write('Total number of cities with restaurants = {}\n'.format(len(city_restaurant_map)))
        fout.write('Total number of categories = {}\n'.format(len(category_map)))
        fout.write('Total number of restaurants without pruning = {}\n'.format(num_businesses_without_stop))
        fout.write('Total number of restaurants after pruning = {}\n'.format(num_businesses_with_stop))
        fout.write('Total number of categories for non-restaurants = {}\n'.format(len(cat_for_non_restaurant_businesses)))
        fout.write('Total number of categories for restaurants = {}\n'.format(len(cat_for_restaurant_businesses)))

    with open('../Logs/not_restaurants.txt', 'w') as fout:
        fout.write('{} businesses are not restaurants.\n'.format(len(businesses_not_restaurants)))
        for business in businesses_not_restaurants:
            fout.write(str(business) + '\n')

    with open('../Logs/restaurants.txt', 'w') as fout:
        fout.write('{} businesses are restaurants.\n'.format(len(restaurants)))
        for business in restaurants:
            fout.write(str(business) + '\n')

    with open('../Logs/cities.txt', 'w') as fout:
        c = Counter(city_restaurant_map)        
        fout.write('{} cities with restaurants.\n'.format(len(city_restaurant_map)))
        for city, count in c.most_common():
            fout.write('{}\t{}\n'.format(city, count))

    with open('../Logs/business_to_city.txt', 'w') as fout:                                
        for bid, city in business_city_map.items():
            fout.write(bid + '::' + city + '\n')


    with open('../Logs/non_restaurant_categories.txt', 'w') as fout:
        c = Counter(cat_for_non_restaurant_businesses)        
        fout.write('{} categories for non-restaurants restaurants.\n'.format(len(cat_for_non_restaurant_businesses)))
        for cat, count in c.most_common():
            fout.write('{}\t{}\n'.format(cat, count))

    with open('../Logs/restaurant_categories.txt', 'w') as fout:
        c = Counter(cat_for_restaurant_businesses)        
        fout.write('{} categories for restaurants restaurants.\n'.format(len(cat_for_restaurant_businesses)))
        for cat, count in c.most_common():
            fout.write('{}\t{}\n'.format(cat, count))

def analyze_reviews(review_file, review_stat_file):
    restaurants = set()   
    with open('../Logs/restaurants.txt', 'r') as fin:
        first_line = True
        for line in fin:
            if first_line == True:
                first_line = False
                continue
            restaurants.add(line[2:24])

    business_city_map = dict()
    with open('../Logs/business_to_city.txt', 'r') as fin:
        for line in fin:
            parts = line.strip().split('::')
            business_city_map[parts[0]] = parts[1]

    # print (str(business_city_map))

    total_num_reviews = 0
    num_reviews_from_non_restaurants = 0
    num_reviews_from_restaurants = 0
    num_reviews_from_las_vegas = 0

    city_freq_map = defaultdict(int)
    restaurant_review_count_map = defaultdict(int)
    words_freq_map = defaultdict(int)    
    
    with open(review_file, 'r') as fin:
        for line in fin:
            review = json.loads(line)

            total_num_reviews += 1

            business_id = review['business_id']
            if business_id not in restaurants:
                num_reviews_from_non_restaurants += 1
            else:
                num_reviews_from_restaurants += 1                
                city = business_city_map[business_id]
                city_freq_map[city] += 1
                if city == 'las vegas':
                    num_reviews_from_las_vegas += 1
                    review_text = review['text']
                    
                    restaurant_review_count_map[business_id] += 1

                    num_words = len(review_text.split())
                    words_freq_map[num_words] += 1
                
            
    
    with open(review_stat_file, 'w') as fout:
        fout.write('Total number of reviews = {}\n'.format(total_num_reviews))
        fout.write('Total number of reviews from non-restaurants = {}\n'.format(num_reviews_from_non_restaurants))
        fout.write('Total number of reviews from restaurants = {}\n'.format(num_reviews_from_restaurants))
        fout.write('Total number of reviews from las vegas restaurants = {}\n'.format(num_reviews_from_las_vegas))
    
    with open('../Logs/reviews_per_las_vegas_restaurant.txt', 'w') as fout:
        c = Counter(restaurant_review_count_map)
        for restaurant, num_reviews in c.most_common():
            fout.write('{}\t{}\n'.format(restaurant, num_reviews))
    
    with open('../Logs/restaurant_reviews_per_city.txt', 'w') as fout:
        c = Counter(city_freq_map)
        for city, num_reviews in c.most_common():
            fout.write('{}\t{}\n'.format(city, num_reviews))
    
    plt.bar(words_freq_map.keys(), words_freq_map.values())
    plt.xlabel('number of words')
    plt.ylabel('number of reviews')
    plt.tight_layout()
    plt.savefig('../Logs/num_words_vs_num_reviews.jpeg')
    
    train_word_vs_review = defaultdict(int)
    with open('../Data/filtered_dataset.json', 'r') as fin:
        for line in fin:
            review = json.loads(line)
            text = review['text']
            num_words = len(text.split())
            train_word_vs_review[num_words] += 1

    test_word_vs_review = defaultdict(int)
    with open('../Data/negative_test_reviews.json', 'r') as fin:
        for line in fin:
            review = json.loads(line)
            text = review['text']
            num_words = len(text.split())
            test_word_vs_review[num_words] += 1

    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    axes[0].bar(train_word_vs_review.keys(), train_word_vs_review.values())    
    axes[1].bar(test_word_vs_review.keys(), test_word_vs_review.values())    

    fig.xlabel('number of words')
    fig.ylabel('number of reviews')
    fig.tight_layout()
    fig.savefig('../Logs/train_test_num_words_vs_num_reviews.jpeg')

    train_word_vs_review = defaultdict(int)
    with open('../Data/filtered_dataset.json', 'r') as fin:
        for line in fin:
            review = json.loads(line)
            text = review['text']
            num_words = len(text.split())
            train_word_vs_review[num_words] += 1

    test_word_vs_review = defaultdict(int)
    with open('../Data/negative_test_reviews.json', 'r') as fin:
        for line in fin:
            review = json.loads(line)
            text = review['text']
            num_words = len(text.split())
            test_word_vs_review[num_words] += 1

    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)
    axes[0].bar(train_word_vs_review.keys(), train_word_vs_review.values())
    axes[0].set_ylabel('number of train reviews')
    axes[0].set_xlabel('number of words in a review')    
    axes[1].bar(test_word_vs_review.keys(), test_word_vs_review.values())    
    axes[1].set_ylabel('number of test reviews')
    axes[1].set_xlabel('number of words in a review')    

    # plt.xlabel('number of words')
    # plt.ylabel('number of reviews')
    plt.tight_layout()
    fig.savefig('../Logs/train_test_num_words_vs_num_reviews.jpeg')


if __name__ == '__main__':
    business_file = '/home/rohan/Documents/yelp_dataset/yelp_academic_dataset_business.json'
    review_file = '/home/rohan/Documents/yelp_dataset/yelp_academic_dataset_review.json'
    stop_file = '../Data/stop_list.txt'
    business_stat_file = '../Logs/business_stat_file.txt'
    review_stat_file = '../Logs/review_stat_file.txt'
    # analyze_business(business_file, stop_file, business_stat_file)
    analyze_reviews(review_file, review_stat_file)

    