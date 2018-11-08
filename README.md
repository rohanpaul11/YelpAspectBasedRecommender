# YelpAspectBasedRecommender
This is the Final Project for COMPSCI - Intro to NLP. We aim to build an aspect based recommendation system using the Yelp dataset.

# Extracting Data
To get the positive reviews, negative train reviews and negative test reviews, simply
execute run.py. 

Before executing, change the path variables in the file to point to the respective 
paths in your specific system. Specifically, you must change the paths of:
business_file
review_file

The output files are created in the Data folder:
positive_reviews.json
negative_train_reviews.json
negative_test_reviews.json
filtered_dataset.json

You can change the number of reviews in the different files by tweaking the parameters
in extract.py:
MIN_REVIEWS_FOR_RESTAURANT
MIN_WORDS_IN_REVIEW
MAX_TEST_REVIEWS