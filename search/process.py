import json

review_file = '../../yelp_academic_dataset_review.json'
with open(review_file,'r') as file, open('trecText','w') as fout:
    limit = 5000
    count = 0
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
        if count==limit:
            break