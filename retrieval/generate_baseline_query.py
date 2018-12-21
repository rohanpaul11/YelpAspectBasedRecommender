import json

class Query:
	def __init__(self, mu, queryList):
		self.mu = mu
		self.queries = queryList

class QueryPair:
	def __init__(self, id, text):
		self.number = id
		self.text = text

def read_test_annotations():
	filename = 'test_data.json'
	test_reviews = {}
	with open(filename, 'r') as f:
		for line in f:
			test_review = json.loads(line)
			if test_review['aspects'] != "" and test_review['query'] != "":
				test_reviews[test_review['review_id']] = test_review
	print(len(test_reviews))
	return test_reviews
	

def generate_query_json(test_reviews, neg_queries):

	queryList = []

	aspects_file = 'test_aspects.txt'
	import ast
	with open(aspects_file, 'r') as f:
		for line in f:
			line = line.strip()
			# print(" line ----------", line)
			review, pos_tuples = line.split(':',1)
			review = review.split()[1]
			pos_tuples = ast.literal_eval(pos_tuples)

			if review in test_reviews and review in neg_queries:
				query_items = []
				if len(pos_tuples)>0:
					for pos_tuple in pos_tuples:
						adj, nsubj, nns = pos_tuple
						query_items.append(nns[0]) 
					query_string = "#combine( " + ' '.join(query_items) + " )"
					query = QueryPair(review, query_string)
					queryList.append(query.__dict__)
					print(' '.join(query_items))
				else:
					query = QueryPair(review, ' ')
					queryList.append(query.__dict__)
					print(' '.join(query_items))
			
			# break
	# print(queryList)
	query = Query(300, queryList)

	# print(json.dumps(query.__dict__))

	with open('base_query.json', 'w') as res:
		res.write(json.dumps(query.__dict__))

def generate_qrels_file(test_reviews):
	with open('test.qrels', 'w') as res:
		for review_id, review in test_reviews.items():
			# print(review_id, review['relevant_reviews'])

			for relevant_review in review['relevant_reviews']:
				res.write(review_id+" 0 "+relevant_review+" 1\n")

def read_negative_aspect_reviews():
	neg_queries = {}
	with open('query.json', 'r') as f:
		queries = json.loads(f.read())["queries"]
		for query in queries:
			neg_queries[query['number']] = query
	return neg_queries


if __name__ == '__main__':
	test_reviews = read_test_annotations()
	neg_queries = read_negative_aspect_reviews()
	generate_query_json(test_reviews, neg_queries)
	generate_qrels_file(test_reviews)
