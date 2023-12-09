import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader
from models.fnn import FNN_LM
from utils.datasets import FNNDataset




'''
we would load the preprocessed text, encoding them into integers and then
extract the (context, target) pairs. Feed the context to the model and 
calculate the loss with the target.

And yes, we can load all data into the memory
'''


def train_FNN(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    '''
    model
    '''
    model = FNN_LM(
            vocab_size=training_config['vocab_size'],
            embedding_dim=training_config['embedding_dim'],
            init_method=training_config['init_method']
            ).to(device)


    #  load
    if training_config.get('model_path_src') is not None:
        model_dict = torch.load(training_config['model_path_src'])
        model.load_state_dict(model_dict)


    '''
    optimizer
    '''
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])


    '''
    loss 
    '''
    criterion = nn.BCELoss().to(device)



    '''
    dataloader
    '''
    dataset_train = FNNDataset(db_path = 'data/BBCnews_processed.db', query = 'SELECT text FROM text_train', encoding_path = 'data/encoding_table.csv', window_width = 2, vocab_size= training_config['vocab_size'])
    dataset_eval  = FNNDataset(db_path = 'data/BBCnews_processed.db', query = 'SELECT text FROM text_eval', encoding_path  = 'data/encoding_table.csv', window_width = 2, vocab_size= training_config['vocab_size'])

    dataloader_train = DataLoader(dataset_train, batch_size = training_config['batch_size'], shuffle = True)
    dataloader_eval  = DataLoader(dataset_eval, batch_size  = training_config['batch_size'], shuffle = True)



    for epoch in range(training_config['num_of_epochs']):
        loss_sum_train = 0
        loss_sum_eval = 0
        #  train loop
        for i, inputs in enumerate(dataloader_train):
            #  inputs = inputs.to(device)
            context, targets = inputs
            context = context.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            probs = model(context)
            loss = criterion(probs, targets)
            loss_sum_train += torch.sum(loss)
            loss.backward()
            optimizer.step()

        #  validation loop
        with torch.no_grad():
            for i, inputs in enumerate(dataloader_eval):
                context, targets = inputs
                context = context.to(device)
                targets = targets.to(device)
                probs = model(context)
                loss = criterion(probs, targets)
                loss_sum_eval += torch.sum(loss)
        
        train_loss = loss_sum_train / len(dataloader_train)
        eval_loss = loss_sum_eval / len(dataloader_eval)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Test Loss: {eval_loss:.6f}')


    model = model.to('cpu')
    torch.save(model.state_dict(), training_config['model_path_dst'])



if __name__ == "__main__":
    training_config = dict()
    training_config['vocab_size']           = 6957
    training_config['embedding_dim']        = 32
    training_config['init_method']          = 'uniform'
    training_config['negative_sample_size'] = 4
    training_config['num_of_epochs']        = 20
    training_config['batch_size']           = 20
    training_config['model_path_dst']       = 'saved_embedding.pth'
    training_config['learning_rate']        = 1e-4
    #  training_config['model_path_src']    = 'saved_embedding.pth'
    train_FNN(training_config)
