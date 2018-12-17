import torch, random
import numpy as np
import json
from torch import nn
import time
from gensim.models import Word2Vec
import re

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
    def forward(self, element):

        num_words = len(element)

        # get embeddings of inputs
        # embs = self.word_embs(element)
        # print(embs.size())

        y = torch.mean(element, 0)

        # get the weights for each word
        d = torch.zeros(num_words, device=self.device)
        for i, word in enumerate(element):
            # t = torch.matmul(torch.transpose(word, 0, 1), M)            
            t = self.M(word)
            # print(t.size(), y.size())
            d[i] = torch.matmul(t, y)
        a = torch.nn.functional.log_softmax(d, dim=0)
        
        # encode the input element        
        z = torch.matmul(torch.transpose(element,0,1), a)

        # prob vector over aspects
        p = torch.nn.functional.softmax(self.W(z), 0)
        
        #reconstructed sentence vector
        r = self.T(p)        
        
        return (z,r)

    # loss function
    def compute_loss(self, Z, R, neg_reviews, _lambda = 0.001):

        J = 0
        for i in range(len(Z)):
            N = 0
            for neg_review in neg_reviews:
                N += torch.matmul(R[i], torch.mean(neg_review, 0))
            J += max(0, 1 - torch.matmul(R[i], Z[i]) + N)

        T_ = self.T.weight
        Tnorm = torch.nn.functional.normalize(T_, dim=1) #p=2 by default
        U = torch.matmul(Tnorm, torch.transpose(Tnorm, 0, 1)) - torch.eye(Tnorm.size()[0], device=self.device)

        loss = J + _lambda * U.norm()
        return loss

