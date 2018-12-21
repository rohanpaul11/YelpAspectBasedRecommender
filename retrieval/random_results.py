import json
import random

def read_test_annotations():
	filename = 'test_data.json'
	test_reviews = []
	with open(filename, 'r') as f:
		for line in f:
			test_review = json.loads(line)
			if test_review['aspects'] != "" and test_review['query'] != "":
				test_reviews.append(test_review['review_id'])
	print(len(test_reviews))
	return test_reviews

def read_all_postitive_reviews():
	review_file = '../aspect/Data/positive_reviews.json'
	ids = []
	with open(review_file,'r') as file, open('trecText','w') as fout:
	    limit = 500000
	    count = 0

	    print_every = 10000
	    for line in file:
	        review = json.loads(line)
	        ids.append(review['review_id'])

	        count+=1

	        if count%print_every==0:
	        	print("Processed {} reviews ".format(count), end='\r')
	        if count==limit:
	        	break

	return ids

	

test_ids = read_test_annotations()
positive_reviews = read_all_postitive_reviews()
print(len(positive_reviews))

with open('random.results','w') as out:
	for test_id in test_ids:
		docs = set()
		for i in range(1000):
			while True:
				pos_review = random.choice(positive_reviews)
				if pos_review in docs:
					print("duplicate")
					continue
				else:
					docs.add(pos_review)
					break
			out.write("{} Q0 {} {} -8.93955539 galago\n".format(test_id, pos_review, i+1))


