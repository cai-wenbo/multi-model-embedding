import torch
import torch.nn as nn
import torch.nn.init as init



'''
the design:
simple recurrent network
input: current text
output: probabilities of the next word
'''
class RNN_LM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, inner_dim):
        super(RNN_LM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim)
        self.inner_dim = inner_dim


        self.rnn = nn.RNN(embedding_dim, inner_dim, batch_first=True)

        self.W = nn.Linear(inner_dim, vocab_size + 2)

        self.softmax = nn.Softmax(dim = 1)


    
    '''
    you can skip the manual model parameters initialization 
    because pytorch has already done it for  us, using 
    Kaiming He initialization
    '''


    '''
    return the probs of the next word in the whold vocab
    '''
    #  text_lists shape = (batch_size, sequence_length)
    def forward(self, text_list):
        #  embedded_inputs shape = (batch_size, sequence_length, embedding_dim)
        embedded_inputs = self.embedding(text_list)
        
        #  inner_states shape = (batch_size, sequence_length, inner_dim)
        inner_states, _ = self.rnn(embedded_inputs)

        #  scores shape = (batch_size, sequence_length, vocab_size + 2)
        scores = self.W(inner_states)

        #  probs shape = (batch_size, sequence_length, vocab_size+ 2)
        probs = self.softmax(scores)

        #  probs shape = (batch_size, sequence_length, vocab_size+ 2)
        return probs
