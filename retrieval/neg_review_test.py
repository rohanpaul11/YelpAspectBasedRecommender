import json

neg_review_file = '../aspect/Data/negative_test_reviews.json'
test_data_file = 'test_data.json'

existing_test_data = {}
with open(test_data_file, 'r') as outfile:
	for line in outfile:
		review = json.loads(line)
		existing_test_data[review['review_id']] = review

with open(neg_review_file,'r') as file, open(test_data_file, 'a') as outfile:
	count = 0
	for line in file:
		review = json.loads(line)
		review_length = len(review['text'].split())
		docs = []
		if review_length<150:
			count += 1
			print("Review number : {}\nReview id : {}\n\n{}\nReview length: {}".format(count, review['review_id'], review['text'], review_length))
			print("---------------------TEST DATA------------------------")
			if review['review_id'] in existing_test_data:
				print(json.dumps(existing_test_data[review['review_id']], indent=4))
			else:
				aspects = input("List of negative aspects: [Example: bad service, high price]\n")
				search_query = input("Search query term: [Example: good serivce cheap price]\n")

				print("Search in galago web UI and enter reviews which you think are relevant to the search query\n")
				for i in range(5):
					doc = input("Relevant review id {} : ".format(i+1))
					docs.append(doc)

				out = {
					"review_id" : review['review_id'],
					"text" : review['text'],
					"aspects" : aspects,
					"query" : search_query,
					"relevant_reviews" : docs
				}
				
				print(json.dumps(out, indent=4))

				save = input("Save these details ? [y/N] : ") or "N"

				if save.lower()=="y":
					outfile.write(json.dumps(out)+"\n")
			print("=======================================================\n\n\n")

