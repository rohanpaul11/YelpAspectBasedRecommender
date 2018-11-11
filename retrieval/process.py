import json

review_file = '../aspect/Data/positive_reviews.json'
with open(review_file,'r') as file, open('trecText','w') as fout:
    limit = 500000
    count = 0

    print_every = 10000
    for line in file:
        review = json.loads(line)
        out = """<DOC>
<DOCNO>
{}
</DOCNO>
<TEXT>
{}
</TEXT>
</DOC>
""".format(review['review_id'], review['text'])
        
        fout.write(out)
        count+=1

        if count%print_every==0:
        	print("Processed {} reviews ".format(count), end='\r')
        if count==limit:
            break