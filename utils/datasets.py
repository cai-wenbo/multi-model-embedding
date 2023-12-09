import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
from torch.utils.data import Dataset
import json
import csv




'''
input: encoding_path 
output: idx2word dict and word2idx dick
'''
def read_encoding_table(encoding_path):
    idx2word = dict()
    word2idx = dict()
    idx2word[0] = 'UNK'


    with open(encoding_path, 'r') as f:
        csv_reader = csv.reader(f)
        
        for row in csv_reader:
            idx = int(row[0])
            word = row[1]
            idx2word[idx] = word
            word2idx[word] = idx

        f.close()

    return idx2word, word2idx


'''
input string
output list of INTEGERs, encoded string, not padded
'''
class Encoder():
    def __init__(self, word2idx):
        self.word2idx = word2idx


    def __call__(self, text):
        words = text.split()
        encoded_words = list()
        for word in words:
            if self.word2idx.get(word) is not None:
                encoded_words.append(self.word2idx[word])
            else:
                encoded_words.append(0)
        return encoded_words




'''
read the proprocessed text from a sqlite db
read the encoding table from csv text 
generate the pairs
'''

class FNNDataset(Dataset):
    def __init__(self, db_path, query, encoding_path, window_width, vocab_size):
        #  get the text
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(query)
        text_list = cursor.fetchall()

        #  get the encoding dict
        idx2word, word2idx = read_encoding_table(encoding_path) 

        encoder = Encoder(word2idx)

        context_list = list()
        targets_list = list()

        #  get the context and  targets
        for text in text_list:
            #  encoding
            encoded_words = encoder(text[0])

            #  padding
            padded_encoded_words = [0] * window_width + encoded_words + [0] * window_width

            for i in range(window_width, len(encoded_words) + window_width):
                context = list()
                target = padded_encoded_words[i]

                for j in range(window_width):
                    context.append(padded_encoded_words[i-window_width+j])
                for j in range(window_width):
                    context.append(padded_encoded_words[i+1+j])

                context_list.append(context)
                targets_list.append(target)


        self.context_list = context_list
        self.targets_list = targets_list
        self.vocab_size   = vocab_size



    def __len__(self):
        return len(self.targets_list)
    

    def __getitem__(self, idx):
        context = self.context_list[idx]
        target  = self.targets_list[idx]


        context_tensor = torch.tensor(context, dtype=torch.int32)
        target_tensor = torch.tensor(F.one_hot(torch.tensor(target), num_classes=self.vocab_size), dtype=torch.float32)
        return context_tensor, target_tensor



