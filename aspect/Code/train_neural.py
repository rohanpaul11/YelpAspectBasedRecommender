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
    # vocab = dict()
    vocab = {'PAD': 0}
    train_reviews = []   
    train_reviews_sents = dict() 
    with open(train_dataset, 'r') as fin:
        for line in fin:
            review = json.loads(line.strip())
            review_id = review['review_id']
            review_text = review['text']
            review_text = re.sub( r'([,.!])([a-zA-Z])', r'\1 \2', review_text)

            sent_text = nltk.sent_tokenize(review_text)
            review_sents = []
            review_embs = []
            review_word_indices = []
            for sentence in sent_text:
                sent = []
                sent_word_indices = []
                words = re.split(r'\W+', sentence.lower())                
                for i, word in enumerate(words):
                    if word in word_embs:
                        if word not in vocab:
                            vocab[word] = len(vocab)
                        sent.append(word_embs[word])
                        sent_word_indices.append(vocab[word])
                        review_embs.append(word_embs[word])
                        review_word_indices.append(vocab[word])
                if len(sent) != 0:
                    review_sents.append((sent, sent_word_indices))
            
            train_reviews.append((review_id, torch.tensor(review_embs, dtype=torch.float, device=device), review_word_indices))
            train_reviews_sents[review_id] = [(torch.tensor(sent[0], dtype=torch.float, device=device), sent[1]) for sent in review_sents]
    
    return vocab, train_reviews, train_reviews_sents

# ==================================================================================================================
def get_embeddings(trained_emb_file):
    embeds = Word2Vec.load(trained_emb_file)        
    embeds.wv.add('PAD', np.zeros(embeds.wv.vector_size)) 
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
        neg_samples.append(train_review_sents[review_id][np.random.randint(0, num_sents)][0])
    
    # print(neg_samples)
    return neg_samples

def perform_kmeans(word_embs, num_aspect_types, device):
    # Kmeans on word embeddings --> initialize T
    kmeans = KMeans(n_clusters=num_aspect_types)
    word_emb_matrix = np.array(word_embs.vectors)  
    kmeans = kmeans.fit(word_emb_matrix)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device).transpose(0,1)
    return cluster_centers
    
# ==================================================================================================================
def train(seed, params, train_reviews, train_review_sents, word_embs, vocab, neural_model_file, log_file):
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

    cluster_centers = perform_kmeans(word_embs, num_aspect_types, device)
    
    net = NeuralAspect(d_word, num_aspect_types, vocab_len, device)
    if torch.cuda.is_available(): net.cuda()
    net.T.weight = torch.nn.Parameter(cluster_centers, requires_grad=True)        
    
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)      
    f = open(log_file, 'w')
    for ep in range(num_epochs):        
        ep_loss = 0

        print('Starting epoch {}...'.format(ep+1))
        start_time = time.time()            
        for start in range(0, num_reviews, batch_size):
            print('\rBatch {}'.format(start//batch_size+1), end='')
            in_batch = train_reviews[start:start + batch_size]
            # print(review_id, review)
            # batch_z = []
            # batch_r = []
            
            if resolution == 'sent':
                maxlen = max([len(sent) for review_id, _, _ in in_batch for sent in train_review_sents[review_id]])
                num_sents = sum([len(train_review_sents[review_id]) for review_id, _, _ in in_batch])             
            else:
                maxlen = max([len(review) for _, review, _ in in_batch])                
            
            bsz = batch_size if resolution != 'sent' else num_sents
            batch = torch.zeros(bsz, maxlen, d_word, device=device)
            
            index = 0
            for review_id, review, _ in in_batch:
                # print(review.size())
                if resolution == 'sent':                    
                    for sent in train_review_sents[review_id]:                        
                        batch[index] = nn.functional.pad(sent, (0, 0, 0, maxlen - len(sent[0])))
                        index += 1
                        # z, r = net(sent)
                        # batch_z.append(z)
                        # batch_r.append(r)
                else:
                    batch[index] = nn.functional.pad(review, (0, 0, 0, maxlen - len(review)))
                    index += 1
                    # z, r = net(review)
                    # batch_z.append(z)
                    # batch_r.append(r)

            batch_z, batch_r = net(batch)
            # print('start', start, batch_z, batch_z, file=f)
            neg_samples = get_neg_samples(train_reviews, train_review_sents, neg_sample_count, resolution)
            batch_loss = net.compute_loss(batch_z, batch_r, neg_samples)
            ep_loss += batch_loss
            optimizer.zero_grad()
            batch_loss.backward()            
            optimizer.step()
            
            # batch_z.clear()
            # batch_r.clear()

        end_time = time.time()            
        diff = end_time - start_time
      
        print("\nLoss after epoch {} = {}".format(ep+1, ep_loss)) 
        print("Time taken for epoch {} = {} hr {} min {} sec".format(ep+1, diff//3600, (diff//60)%60, diff%60))              

        print('Saving the model...')
        torch.save(net.state_dict(), neural_model_file)

    print('Training complete!!')
    f.close()

# ==================================================================================================================
def load_saved_net(neural_model_file, vocab, params):
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
    for review_id, review, indices in reviews:
        # review = torch.tensor([word_emb for word_emb, _ in review_with_index], device=device)

        aspects = set()
        weight_aspect_pairs = []
        sents = [(review, indices)] if resolution != 'sent' else review_sents[review_id]
        for sent, word_indices in sents:
            y = torch.mean(sent, 0)

            # get the weights for each word
            d = torch.zeros(len(sent), device=device)
            for i, word in enumerate(sent):
                # t = torch.matmul(torch.transpose(word, 0, 1), M)       
                if torch.cuda.is_available(): M.cuda()
                t = M(word)
                # print(t.size(), y.size())
                d[i] = torch.matmul(t, y)
            a = nn.functional.softmax(d, dim=0)
            # print(a.size())
            # GET TOP ASPECTS FROM HERE
            _, top_indices = torch.topk(a, len(a)//int(np.sqrt(len(a))), largest=False)            
            
            for index in top_indices:                
                if resolution == 'sent':
                    aspect_word = idx_to_word[word_indices[index.item()]]
                else:
                    aspect_word = idx_to_word[indices[index.item()]]
                if aspect_word == 'PAD': continue
                if aspect_word not in aspects:
                    aspects.add(aspect_word)
                    weight_aspect_pairs.append((a[index].item(), aspect_word))
        
        aspects_per_review.append((review_id, weight_aspect_pairs))

    return aspects_per_review

# ==================================================================================================================
def retrieve_and_save_aspects(vocab, train_reviews, train_review_sents, saved_net, num_aspects, device, aspect_file, resolution):

    idx_to_word = dict({(v, k) for k, v in vocab.items()})
    aspects_per_review = retrieve_aspects(train_reviews, train_review_sents, saved_net.M, num_aspects, idx_to_word, device, resolution)
    with open(aspect_file, 'w') as fout:
        for review_id, aspects in aspects_per_review:
            fout.write('{}:{}\n'.format(review_id, aspects))
            # fout.write('{}:{}\n'.format(review_id, [word for weight, word in aspects]))

# ==================================================================================================================
if __name__ == '__main__':
    seed = 65537
    train_dataset = '../Data/filtered_dataset.json'
    test_dataset = '../Data/test_data.json'
    # train_dataset = '../Data/toy_sample.json'
    trained_embedding_file = '../Data/model_file'
    neural_model_file = '../Data/trained_model'
    aspect_file = '../Data/neural_aspects.txt'
    log_file = '../Logs/neural_log.log'

    params = {
        'd_word' : 200 ,
        'num_aspect_types' : 14,
        'num_epochs' : 15,
        'batch_size' : 500,
        'neg_sample_count' : 20,
        'resolution' : 'sent',
        'num_aspects' : 5,
        'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    word_embs = get_embeddings(trained_embedding_file)
    vocab, train_reviews, train_review_sents = compute_vocab_and_get_reviews(train_dataset, word_embs, params['device'])    
    # print(vocab)
    train(seed, params, train_reviews, train_review_sents, word_embs, vocab, neural_model_file, log_file)
    
    saved_net = load_saved_net(neural_model_file, vocab, params)

    retrieve_and_save_aspects(vocab, train_reviews, train_review_sents, saved_net, params['num_aspects'], params['device'], aspect_file, params['resolution'])

    test_vocab, test_reviews, test_review_sents = compute_vocab_and_get_reviews(test_dataset, word_embs, params['device'])
    retrieve_and_save_aspects(test_vocab, test_reviews, test_review_sents, saved_net, params['num_aspects'], params['device'], aspect_file, params['resolution'])
    