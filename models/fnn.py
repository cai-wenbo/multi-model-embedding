import torch
import torch.nn as nn
import torch.nn.init as init



''' 
This model is quite similar to the original CBoW model,
except that it doesn't use the negative samples, instead 
it would calculate the loss across the entire vocab! 
You can view the linear layer as the target-side embedding 
whild the embedding layer as the src-side embedding. Very simple!

Read this article: 

input: (context)
output: (probabilities)
where v_c is the average of the context vectors
'''
class FNN_LM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, init_method = None):
        super(FNN_LM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.init_params(init_method)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim = 1)


    
    '''
    you can skip init because pytorch has already done it for 
    us, using Kaiming He initialization
    '''
    def init_params(self, init_method):
        if init_method is not None:
            if init_method == 'uniform':
                init.uniform_(self.embedding.weight.data, a = -1, b = 1)
            elif init_method == 'normal':
                init.normal_(self.embedding.weight.data, mean = 0, std = 1)

        


    '''
    return the probs of the whole vocab
    '''
    def forward(self, context):
        #  context shape = (batch_size, context_size)
        #  targets shape = (batch_size, target_size)
        

        # embedded_context shape = (batch_size, context_size, embedding_dim)
        #  v_c shape = (batch_size, embedding_dim)
        embedded_context = self.embedding(context)
        v_c = torch.sum(embedded_context, dim = 1) / embedded_context.shape[1]

        #  scores shape = (batch_size, vocab_size)
        scores = self.fc(v_c)

        #  probs shape = (batch_size, vocab_size)
        probs = self.softmax(scores)

        return probs



    def get_cross_scores(self, src_tensor, trgs_tensor):
        embedded_src = self.embedding(src_tensor)
        embedded_targets = self.embedding(trgs_tensor)

        scores = torch.matmul(embedded_src.unsqueeze(1), embedded_targets.transpose(-2,-1)).squeeze(dim=1)
        return scores

