import torch, random
import numpy as np
import json
from torch import nn
import time
from gensim.models import Word2Vec
import re
from sklearn.cluster import KMeans

class NeuralAspect(nn.Module):
    def __init__(self, d_word, num_aspect_types, len_voc, device):
        super(NeuralAspect, self).__init__()
        self.d_word = d_word
        self.num_aspect_types = num_aspect_types
        self.len_voc = len_voc
        self.device = device

        # embeddings
        # self.word_embs = nn.Embedding(len_voc, d_word).from_pretrained(pretrained, freeze=True)
        
        # layer used for encoding word embs
        self.M = nn.Linear(d_word, d_word)
        
        # layer used for learning aspect weights
        self.W = nn.Linear(d_word, num_aspect_types, bias=True)
        
        # output layer
        self.T = nn.Linear(num_aspect_types, d_word)


    # perform forward propagation of NeuralAspect model
    # element may be a sentence or a review
    def forward(self, elements):

        bsz, num_words, _ = elements.size()
        # num_words = len(element)

        # get embeddings of inputs
        # embs = self.word_embs(element)
        # print(embs.size())

        y = torch.mean(elements, 1)
        # print('y=',y.size())
        # y = torch.mean(element, 0)
        # get the weights for each word
        
        # for i, word in enumerate(element):
        #     # t = torch.matmul(torch.transpose(word, 0, 1), M)            
        #     t = self.M(word)
        #     # print(t.size(), y.size())
        #     d[i] = torch.matmul(t, y)
        # a = nn.functional.softmax(d, dim=0)
        
        t = self.M(elements)
        # print('t=',t.size())
        d = torch.bmm(t, y.unsqueeze(-1))
        # print('d=',d.size())
        a = nn.functional.softmax(d, dim=1)
        # print(a)
        # print('a=',a.size())
        z = torch.bmm(a.transpose(1,2), elements)
        # print('z=',z.size())
        p = nn.functional.softmax(self.W(z), 0)        
        # print(p)
        # print('p=',p.size())
        # # encode the input element        
        # z = torch.matmul(torch.transpose(element,0,1), a)

        # prob vector over aspects
        # p = torch.nn.functional.softmax(self.W(z), 0)
        
        #reconstructed sentence vector
        r = self.T(p)
        # print(r.size())        
        
        return (z,r)

    # loss function
    def compute_loss(self, Z, R, neg_reviews, _lambda = 1):

        J = 0
        for i in range(len(Z)):
            N = 0
            for neg_review in neg_reviews:
                N += torch.dot(R[i].view(-1), torch.mean(neg_review, 0).view(-1))
            J += max(0, 1 - torch.dot(R[i].view(-1), Z[i].view(-1)) + N)

        T_ = self.T.weight
        Tnorm = torch.nn.functional.normalize(T_, dim=1) #p=2 by default
        # print(Tnorm)
        U = torch.matmul(Tnorm, torch.transpose(Tnorm, 0, 1)) - torch.eye(Tnorm.size()[0], device=self.device)

        # print('J=',J, 'Tnorm=', Tnorm)
        loss = J + _lambda * U.norm()
        return loss

