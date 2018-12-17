import torch, random
import numpy as np
import json
from torch import nn
import time
from gensim.models import Word2Vec
import nltk
import re
from neural import *

# ==================================================================================================================
def compute_vocab_and_get_reviews(train_dataset, word_embs, device):
    vocab = dict()
    train_reviews = []   
    train_reviews_sents = dict() 
    with open(train_dataset, 'r') as fin:
    # with open("filtered_dataset.json", 'r') as fin:
        for line in fin:
            review = json.loads(line.strip())
            review_id = review['review_id']
            review_text = review['text']
            review_text = re.sub( r'([,.!])([a-zA-Z])', r'\1 \2', review_text)

            sent_text = nltk.sent_tokenize(review_text)
            review_sents = []
            review_embs = []
            # word_indices = []
            for sentence in sent_text:
        #       tokenized_text = nltk.word_tokenize(sentence.lower())
        #       print(sentence.lower())
                sent = []
                words = re.split(r'\W+', sentence.lower())                
                for i, word in enumerate(words):
                    if word in word_embs:
                        if word not in vocab:
                            vocab[word] = len(vocab)
                        # print(type(word_embs[word]))
                        # print(type(vocab[word]))
                        sent.append(word_embs[word])
                        review_embs.append(word_embs[word])
                        # word_indices.append(vocab[word])
                if len(sent) != 0:
                    review_sents.append(sent)
            
            train_reviews.append((review_id, torch.tensor(review_embs, device=device)))
            train_reviews_sents[review_id] = [torch.tensor(sent, device=device) for sent in review_sents]
    
    return vocab, train_reviews, train_reviews_sents

# ==================================================================================================================
def get_embeddings(trained_emb_file):
    embeds = Word2Vec.load(trained_emb_file)
    return embeds.wv
    
# ==================================================================================================================
def get_neg_samples(train_reviews, train_review_sents, k, resolution):
    neg_review_indices = np.random.choice(len(train_reviews), size=k)   
    # print(neg_review_indices) 

    if resolution != 'sent':
        return [train_reviews[neg_review_idx][1] for neg_review_idx in neg_review_indices]

    neg_samples = []
    for neg_review_index in neg_review_indices:
        review_id = train_reviews[neg_review_index][0]
        num_sents = len(train_review_sents[review_id])
        neg_samples.append(train_review_sents[review_id][np.random.randint(0, num_sents)])
    
    return neg_samples

# ==================================================================================================================
def train(seed, params, train_reviews, train_review_sents, word_embs, vocab, neural_model_file):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)    

    vocab_len = len(vocab)
    num_reviews = len(train_reviews)
    assert(len(train_review_sents) == num_reviews)

    d_word = params['d_word'] 
    num_aspect_types = params['num_aspect_types']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    neg_sample_count = params['neg_sample_count']
    resolution = params['resolution']
    device = params['device']

    net = NeuralAspect(d_word, num_aspect_types, vocab_len, device)
    if torch.cuda.is_available(): net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)  
    
    
    for ep in range(num_epochs):        
        ep_loss = 0

        print('Starting epoch {}...'.format(ep+1))
        start_time = time.time()            
        for start in range(0, num_reviews, batch_size):
            print('Batch {}\r'.format(start//batch_size+1), end='')
            in_batch = train_reviews[start:start + batch_size]
            # print(review_id, review)
            batch_z = []
            batch_r = []
            for review_id, review in in_batch:
                # print(review.size())
                if resolution == 'sent':
                    for sent in train_review_sents[review_id]:
                        # print(sent, sent.size())
                        z, r = net(sent)
                        batch_z.append(z)
                        batch_r.append(r)
                else:
                    z, r = net(review)
                    batch_z.append(z)
                    batch_r.append(r)

            neg_samples = get_neg_samples(train_reviews, train_review_sents, neg_sample_count, resolution)
            ep_loss += net.compute_loss(batch_z, batch_r, neg_samples)
            optimizer.zero_grad()
            ep_loss.backward(retain_graph=True)            
            optimizer.step()
            
            batch_z.clear()
            batch_r.clear()

        end_time = time.time()            
        diff = end_time - start_time
        # if ((ep+1) % 5) == 0 or (ep+1 == num_epochs): 
        print("\nLoss after epoch {} = {}".format(ep+1, ep_loss)) 
        print("Time taken for epoch {} = {} hr {} min {} sec".format(ep+1, diff//3600, (diff//60)%60, diff%60))              

        print('Saving the model...')
        torch.save(net.state_dict(), neural_model_file)

    print('Training complete!!')

# ==================================================================================================================
def load_saved_net(neural_model_file, params):
    d_word = params['d_word'] 
    num_aspect_types = params['num_aspect_types']
    device = params['device']

    print('Loading saved neural network...')
    saved_net = NeuralAspect(d_word,num_aspect_types,len(vocab), device)
    saved_net.load_state_dict(torch.load(neural_model_file, map_location=lambda storage, loc: storage))    

    print('Loading complete!')
    # print(saved_net.state_dict())

    return saved_net

# ==================================================================================================================
def retrieve_aspects(reviews, review_sents, M, num_aspects, idx_to_word, device, resolution):
    aspects_per_review = []
    for review_id, review in reviews:
        # review = torch.tensor([word_emb for word_emb, _ in review_with_index], device=device)

        aspects = []
        sents = [review] if resolution != 'sent' else review_sents[review_id]
        for sent in sents:
            y = torch.mean(sent, 0)

            # get the weights for each word
            d = torch.zeros(len(sent), device=device)
            for i, word in enumerate(sent):
                # t = torch.matmul(torch.transpose(word, 0, 1), M)       
                if torch.cuda.is_available(): M.cuda()
                t = M(word)
                # print(t.size(), y.size())
                d[i] = torch.matmul(t, y)
            a = torch.nn.functional.log_softmax(d, dim=0)
            # print(a.size())
            _, top_indices = torch.topk(a, min(num_aspects, len(a)))            
            
            for index in top_indices:
                aspects.append(idx_to_word[index.item()])
        
        aspects_per_review.append((review_id, aspects))

    return aspects_per_review

# ==================================================================================================================
def retrieve_and_save_aspects(vocab, train_reviews, train_review_sents, saved_net, num_aspects, device, aspect_file, resolution):

    idx_to_word = dict({(v, k) for k, v in vocab.items()})
    aspects_per_review = retrieve_aspects(train_reviews, train_review_sents, saved_net.M, num_aspects, idx_to_word, device, resolution)
    with open(aspect_file, 'w') as fout:
        for review_id, aspects in aspects_per_review:
            fout.write('{}:{}\n'.format(review_id, aspects))

# ==================================================================================================================
if __name__ == '__main__':
    seed = 65537
    # train_dataset = '../Data/filtered_dataset.json'
    train_dataset = '../Data/toy_sample.json'
    trained_embedding_file = '../Data/model_file'
    neural_model_file = '../Data/trained_model'
    aspect_file = '../Data/neural_aspects.txt'
    
    params = {
        'd_word' : 200 ,
        'num_aspect_types' : 14,
        'num_epochs' : 15,
        'batch_size' : 16,
        'neg_sample_count' : 20,
        'resolution' : 'sent',
        'num_aspects' : 5,
        'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    word_embs = get_embeddings(trained_embedding_file)
    vocab, train_reviews, train_review_sents = compute_vocab_and_get_reviews(train_dataset, word_embs, params['device'])    

    train(seed, params, train_reviews, train_review_sents, word_embs, vocab, neural_model_file)
    
    saved_net = load_saved_net(neural_model_file, params)

    retrieve_and_save_aspects(vocab, train_reviews, train_review_sents, saved_net, params['num_aspects'], params['device'], aspect_file, params['resolution'])

    