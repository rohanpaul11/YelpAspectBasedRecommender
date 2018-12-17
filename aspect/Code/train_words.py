import json
import re
import nltk
nltk.download('punkt')
from gensim.models import Word2Vec

def train_word_embs(train_dataset, word_embs_file):
    sentences = []

    with open(train_dataset) as f:
        for line in f:
            review = json.loads(line.strip())
            review_id = review['review_id']
            review_text = review['text']
            review_text = re.sub( r'([,.!])([a-zA-Z])', r'\1 \2', review_text)

            sent_text = nltk.sent_tokenize(review_text)
            for sentence in sent_text:
                # tokenized_text = nltk.word_tokenize(sentence.lower())
                # print(sentence.lower())
                words = re.split(r'\W+', sentence.lower())
                processed_sent = [word.lower() for word in words if word]
                if processed_sent:
                    sentences.append(processed_sent)
        
    print(len(sentences))

    # define training data
    # train model
    model = Word2Vec(sentences, size=200, window=5, min_count=1)
    model.save(word_embs_file)
    print(model)

if __name__ == '__main__':
    train_dataset = '../Data/filtered_dataset.json'
    trained_embedding_file = '../Data/model_file'
    train_word_embs(train_dataset, trained_embedding_file)
