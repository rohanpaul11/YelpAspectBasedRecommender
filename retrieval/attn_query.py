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
	# print(len(test_reviews))
	return test_reviews
	

def generate_query_json(test_reviews):

	queryList = []

	aspects_file = 'extracted_aspects.json'

	with open(aspects_file, 'r') as f:
		neg_reviews = f.read()
		# print(neg_reviews)

		neg_reviews = json.loads(neg_reviews)

		for neg_review, aspects in neg_reviews.items():
			print(aspects)
			if neg_review in test_reviews:
				query_items = []
				for neg_aspects in aspects[:1]:
					query_items.append(neg_aspects[1])
				query_string = "#combine( " + ' '.join(query_items) + " )"
				query = QueryPair(neg_review, query_string)
				queryList.append(query.__dict__)


		print("Total number of queries {}".format(len(queryList)))
		query = Query(300, queryList)

	with open('attn_query.json', 'w') as res:
		res.write(json.dumps(query.__dict__))

	# 	for line in f:
	# 		line = line.strip()
	# 		review, pos_tuples = line.split(':')
	# 		review = review.split()[1]
	# 		pos_tuples = ast.literal_eval(pos_tuples)

	# 		if review in test_reviews:
	# 			query_items = []
	# 			if len(pos_tuples)>0:
	# 				for pos_tuple in pos_tuples:
	# 					adj, nsubj, nns = pos_tuple
	# 					query_items.append(nns[0]) 
	# 				query_string = "#combine( " + ' '.join(query_items) + " )"
	# 				query = QueryPair(review, query_string)
	# 				queryList.append(query.__dict__)
	# 				print(' '.join(query_items))
	# 			else:
	# 				query = QueryPair(review, ' ')
	# 				queryList.append(query.__dict__)
	# 				print(' '.join(query_items))
			
	# 		# break
	# # print(queryList)
	# query = Query(1000, queryList)

	# # print(json.dumps(query.__dict__))

	# with open('query.json', 'w') as res:
	# 	res.write(json.dumps(query.__dict__))

def generate_qrels_file(test_reviews):
	with open('test.qrels', 'w') as res:
		for review_id, review in test_reviews.items():
			# print(review_id, review['relevant_reviews'])

			for relevant_review in review['relevant_reviews']:
				res.write(review_id+" 0 "+relevant_review+" 1\n")



if __name__ == '__main__':
	test_reviews = read_test_annotations()
	generate_query_json(test_reviews)
	# generate_qrels_file(test_reviews)