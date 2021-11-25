from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from torch import nn


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def change_numbers_in_graph(filename="test_files/res1_hp_temp_kg_SMALL.ttl", new_fn=None):
    if new_fn == None:
        new_fn = filename[:-4]+'_CHANGED.ttl'
    new_lines = []
    with open(filename) as ttl_data:
        for line in ttl_data.readlines():
            if 'hasValue' in line: #take the value out of the line (and keep the prefix and suffix for later)
                part1 = line[:19]
                value = line[19:-3]
                part3 = line[-3:]
                if value[-3:] == '+00': #change scientific notation into regular notation (only these three were relevant)
                    value = value[:-4]
                elif value[-3:] == '+01':
                    value = str(float(value[:-4]) * 10)
                elif value[-3:] == '-01':
                    value = str(float(value[:-4]) / 10)
                else:
                    print("STRANGE NUMBER: ", value) #catch if an "out of bounds" scientific notation has been encountered
                    exit()
                new_value = '"'+value.replace('.', ',')+'"' #replace '.' with ',' to make sure the string stays a string (and not a float)
                for i in range(1,7): #remove trailing digits caused by conversion
                    if new_value.endswith("000"+str(i)+"\""):
                        new_value = new_value[:new_value.find("000")]+'"'
                new_lines.append(part1+new_value+part3)
            else:
                new_lines.append(line) #append non-value lines to the new file

    with open(new_fn, 'w') as ttl_data:
        for line in new_lines:
            ttl_data.write(line)


#change numbers to work
def change_numbers_in_entities(entities_fn="test_files/res1_entities_SMALL.tsv", graph_fn="test_files/res1_hp_temp_kg_SMALL_CHANGED.ttl", new_entities_fn=None):
    if new_entities_fn == None:
        new_entities_fn = entities_fn[:-4]+'_CHANGED.tsv'
    ttl_entities = get_entities_from_ttl(graph_fn)
    new_lines = []
    collected_powers = []
    skipped_entities = {'double': 0, 'not_in_graph': 0}
    
    with open(entities_fn) as tsv_data:
        new_lines.append(tsv_data.readline())
        for line in tsv_data.readlines():
            line = line.split('\t')
            power_usage = line[0]
            temp = line[1]
            new_power = str(power_usage).replace('.', ',')
            if new_power in ttl_entities: #check if the entity exists in the graph
                if not new_power in collected_powers: #only add each entity once, to not break one-to-one embedding
                    new_lines.append('\t'.join([new_power,temp]))
                    collected_powers.append(new_power)
                else:
#                     print("skipped entity: DOUBLE ENTITY")
                    skipped_entities['double'] += 1
            else:
#                 print("skipped entity: not available in graph")
                skipped_entities['not_in_graph'] += 1
    print('skipped', skipped_entities['double'], 'DOUBLE entities and', skipped_entities['not_in_graph'], 'UNAVAILABLE entities.')
    with open(new_entities_fn, 'w') as tsv_data:
        for line in new_lines:
            tsv_data.write(line)

def get_entities_from_ttl(ttl_filename):
    new_values_ttl = []

    with open(ttl_filename) as ttl_data:
        for line in ttl_data.readlines():
            if 'hasValue' in line:
                value = line[19:-3]
                new_values_ttl.append(value[1:-1])
    return new_values_ttl


import pandas as pd


def make_embeddings(entities_fn="test_files/res1_entities_SMALL_CHANGED.tsv", kg_fn="test_files/res1_hp_temp_kg_SMALL_CHANGED.ttl", new_entities_fn=None, entities_column_name="power_usage", reverse=False):
    if new_entities_fn == None:
        new_entities_fn = entities_fn[:-4]+'_embeddings.tsv'
    data = pd.read_csv(entities_fn, sep="\t")
    
    entities = [entity for entity in data[entities_column_name]]
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=1),
        walkers=[RandomWalker(4, 10, with_reverse=reverse, n_jobs=8, md5_bytes=None)],
        verbose=1
    )
    kg = KG(location=kg_fn)
    embeddings, literals = transformer.fit_transform(kg, entities)

    new_emb = []
    for embedding in embeddings:
        new_emb.append(embedding.tolist())

    data[entities_column_name+'_emb'] = new_emb
    data.to_csv(new_entities_fn, sep="\t", index=False)


def get_walks(entities_fn="test_files/res1_entities_SMALL_CHANGED.tsv", kg_fn="test_files/res1_hp_temp_kg_SMALL_CHANGED.ttl", entities_column_name="power_usage"):
    data = pd.read_csv(entities_fn, sep="\t")
    
    entities = [entity for entity in data[entities_column_name]]
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=1),
        walkers=[RandomWalker(4, 10, with_reverse=True, n_jobs=8, md5_bytes=None)],
        verbose=2
    )
    kg = KG(location=kg_fn)

    print(transformer.get_walks(kg,entities))


#based on: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
def show_hp_emb_temp_values(power_usage_emb, tempC_average):
    print(power_usage_emb, tempC_average)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        hp, temp = sample['power_usage_emb'], sample['tempC_average']
        return {'power_usage_emb': torch.from_numpy(hp),
                'tempC_average': torch.from_numpy(temp)}


class HP_emb_TempDataset(Dataset):
    """Dataset containing the Heat Pump power consumption values and the temperature at that time."""

    def __init__(self, tsv_file='res1_entities_embeddings.tsv', train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the tsv file with two columns, hp consumption and temp.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        tsv_file = pd.read_csv(tsv_file, sep="\t")
        if train:
            self.hptempvalues = tsv_file[:int(len(tsv_file)*0.8)].reset_index() #[power_usage, tempC_average]
        else:
            self.hptempvalues = tsv_file[int(len(tsv_file)*0.8):].reset_index() #[power_usage, tempC_average]

        self.transform = transform

    def __len__(self):
        return len(self.hptempvalues)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("idx:", idx)
        sample = {'power_usage_emb': np.array([float(x.strip(' []')) for x in self.hptempvalues['power_usage_emb'][idx].split(',')]), 'tempC_average': np.array([float(self.hptempvalues['tempC_average'][idx])]) }
        if self.transform:
            sample = self.transform(sample)

        return sample

# Helper function to show a batch
def show_hp_emb_temp_batch(sample_batched):
    """Show power usage of heat pump and temp"""
    power_batch, temp_batch = \
            sample_batched['power_usage_emb'], sample_batched['tempC_average']
    batch_size = len(power_batch)

    print('Batch from dataloader')
    for i in range(batch_size):
        print(power_batch[i,:], temp_batch[i,:])

# if __name__ == '__main__':
#     hp_temp_dataset = HP_emb_TempDataset()

#     for i in range(len(hp_temp_dataset)):
#         sample = hp_temp_dataset[i]

#         print(i, sample['power_usage_emb'].shape, sample['tempC_average'].shape)
#         show_hp_emb_temp_values(**sample)

#         if i > 3:
#             break
#     print(len((hp_temp_dataset[0]['power_usage_emb']).tolist()))


def perform_prediction(dataset_fn = 'res1_entities_embeddings.tsv', results_fn = 'results.txt'):
    training_data = HP_emb_TempDataset(tsv_file=dataset_fn, train=True, transform=ToTensor())
    test_data = HP_emb_TempDataset(tsv_file=dataset_fn, train=False, transform=ToTensor())

    batch_size = 4

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for sample in test_dataloader:
        X = sample['power_usage_emb']
        y = sample['tempC_average']
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    # exit()

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(100, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    model = model.float()
    print(model)


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, sample in enumerate(dataloader):
            X = sample['power_usage_emb']
            y = sample['tempC_average']
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X.float())
            loss = loss_fn(pred.float(), y.float())


            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # print("x", X[0], "y", torch.round(y[0]), "pred", torch.round(pred[0]))
                # print("x", X[0], "y", torch.round(y[0]), "pred", torch.round(pred[0]))


    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, within_2, within_5 = 0, 0, 0
        with torch.no_grad():
            for sample in dataloader:
                X = sample['power_usage_emb']
                y = sample['tempC_average']
                X, y = X.to(device), y.to(device)
                pred = model(X.float())
                test_loss += loss_fn(pred, y).item()
                # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                for i in range(0, len(y)):
                    if abs(int(y[i].item()) - int(pred[i].item())) < 3:
                        within_2 += 1
                        within_5 += 1
                    elif abs(int(y[i].item()) - int(pred[i].item())) < 6:
                        within_5 += 1
        test_loss /= num_batches
        within_2 = within_2 / size
        within_5 = (within_2 + within_5) / size
        # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print(f"Test Error: \n accurate within 2 degrees: {(100*within_2):>0.1f}%, \n accurate within 5 degrees: {(100*within_5):>0.1f}%, \n Avg loss: {test_loss:>8f} \n")
        return within_2


    epochs = 15
    model = model.float()
    res = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        res.append(test(test_dataloader, model, loss_fn))
    print("Done!")
    print(res)
    with open(results_fn, 'w') as results_file:
        for r in res:
            results_file.write(str(r)+'\n')


def pipeline():
    # graph_name = "res1_hp_temp_kg.ttl"
    graph_name = "test_files/res1_hp_temp_kg_SMALL.ttl"
    entities_name ="test_files/res1_entities_SMALL.tsv"

    #the names of the changed graph and entities files
    graph_c = "res1_g_final_test.ttl"
    entities_c = "res1_e_final_test.tsv"

    # #updated entities file with embeddings
    entities_emb = "res1_emb_final_test.tsv"

    change_numbers_in_graph(filename=graph_name, 
                            new_fn=graph_c)
    change_numbers_in_entities(entities_fn=entities_name, 
                               graph_fn=graph_c, 
                               new_entities_fn=entities_c)

    make_embeddings(entities_fn=entities_c, 
                    kg_fn=graph_c, 
                    new_entities_fn=entities_emb,
                    entities_column_name="power_usage",reverse=True) #CHANGE REVERSE TO FALSE?!

    perform_prediction(dataset_fn=entities_emb, results_fn="results_test1.txt")

if __name__ == '__main__':
    pipeline()