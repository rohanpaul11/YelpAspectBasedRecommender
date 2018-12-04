import torch, random
import numpy as np
import json
from torch import nn

class NeuralAspect(nn.Module):
    def __init__(self, d_word, num_aspects, len_voc):
        super(NeuralAspect, self).__init__()
        self.d_word = d_word
        self.num_aspects = num_aspects
        self.len_voc = len_voc
        # self.M = torch.ones(d_word, d_word)

        # embeddings
        self.word_embs = nn.Embedding(len_voc, d_word)

        # layer used for encoding word embs
        self.M = nn.Linear(d_word, d_word)

        # layer used for learning aspect weights
        self.W = nn.Linear(d_word, num_aspects, bias=True)
        
        # output layer
        self.out = nn.Linear(num_aspects, d_word)

    # perform forward propagation of NeuralAspect model
    def forward(self, review):

        num_words = review.size()

        # get embeddings of inputs
        embs = self.word_embs(review)
        # print(embs.size())

        y = torch.mean(embs, 0)

        # get the weights for each word
        d = torch.zeros(num_words)
        for i, word in enumerate(embs):
            # t = torch.matmul(torch.transpose(word, 0, 1), M)            
            t = self.M(word)
            # print(t.size(), y.size())
            d[i] = torch.matmul(t, y)
        a = torch.nn.functional.log_softmax(d, dim=0)

        # encode the input review        
        z = torch.matmul(torch.transpose(embs,0,1), a)

        # prob vector over aspects
        p = torch.nn.functional.softmax(self.W(z), 0)
        
        #reconstructed sentence vector
        r = self.out(p)        
        
        return (z,r)

    # loss function
    def compute_loss(self, Z, R, _lambda = 0.001):
        J = 0
        for i in range(len(Z)):
            J += max(0, 1 - torch.matmul(R[i], Z[i]))

        T = self.out.weight
        Tnorm = torch.nn.functional.normalize(T, dim=1) #p=2 by default
        U = torch.matmul(Tnorm, torch.transpose(Tnorm, 0, 1)) - torch.eye(Tnorm.size()[0])

        loss = J + _lambda * U.norm()
        return loss

   
if __name__ == '__main__':
    torch.manual_seed(1111)
    random.seed(1111)
    np.random.seed(1111)
    
    vocab = dict()
    train_reviews = []
    with open('../Data/toy_sample.json', 'r') as fin:
        for line in fin:
            review = json.loads(line.strip())
            review_id = review['review_id']
            review_text = review['text']
            words = review_text.lower().split()
            words_as_indices = torch.LongTensor(len(words))
            for i, word in enumerate(words):
                if word not in vocab:
                    vocab[word] = len(vocab)
                words_as_indices[i] = vocab[word]
            train_reviews.append((review_id, words_as_indices))
    
    vocab_len = len(vocab)
    num_aspects = 4
    num_epochs = 100
    batch_size = 5
    net = NeuralAspect(10, num_aspects, vocab_len)    
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)

    output = dict() 
    for ep in range(num_epochs):        
        ep_loss = 0
        ep_out = dict()

        Z = []
        R = []
        for start in range(0, len(train_reviews), batch_size):
            in_batch = train_reviews[start:start + batch_size]
            # print(review_id, review)
            batch_z = []
            batch_r = []
            for review_id, review in in_batch:
                z, r = net(review)
                batch_z.append(z)
                batch_r.append(r)
                
                Z.append(z)
                R.append(r)
                ep_out[review_id] = (z, r)

            
            ep_loss += net.compute_loss(Z, R)
            optimizer.zero_grad()
            ep_loss.backward(retain_graph=True)            
            optimizer.step()

        if ((ep+1) % 10) == 0 or (ep+1 == num_epochs): 
            print("Loss after epoch {} = {}".format(ep+1, ep_loss))
        if ep == num_epochs - 1:
            output = ep_out
    
    with open('../Data/neural_output.txt', 'w') as fout:
        for review_id, sent_and_aspect in output.items():
            print(sent_and_aspect)
            fout.write('{}:{}\n'.format(review_id, sent_and_aspect))
    
    
                
            

            
            