"""
Main train loop
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import yaml
from data import Data
from pascal_context_data import PascalContextData
import torch.optim as optim
from model import SimpleFCN
import pdb
# reads the config file, returns appropriate instantiation of Data class
def _dataset_factory(cfg_file) -> Data:
    f = open(cfg_file, 'r')
    cfg = yaml.load(f)
    f.close()
    if cfg['name'] == 'pascal_context':
        return PascalContextData(cfg)
    else:
        print("Dataset name not matched")
        exit(-1)

# reads the config file, returns appropriate instantiation of pytorch Module class
def _model_factory(cfg_file) -> torch.nn.Module:
    f = open(cfg_file, 'r')
    cfg = yaml.load(f)
    f.close()
    if cfg['name'] == 'simple_fcn':
        return SimpleFCN(cfg)
    else:
        print("Model name not matched")
        exit(-1)

        
def train(model, data):
    # train the model
    model.cuda()
    num_batches = data.get_num_batches()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        data.shuffle() # shuffle the dataset
        for iter in range(num_batches):
            # inputs sholud be N x 224 x 224 x 3
            inputs, labels = data.get_batch(iter)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # labels and outputs should be N x 28 x 28 x 459
            labels_reshaped = labels.view(-1)

            outputs = model(inputs)
            outputs_reshaped = outputs.view(-1,459)
            loss = criterion(outputs_reshaped, labels_reshaped)
            loss.backward()
            optimizer.step()
            print(loss.data[0])

            
if __name__ == '__main__':    
    dataset_cfg = sys.argv[1]
    # does nothing for now
    model_cfg = sys.argv[2]
    # load the appropriate dataset into a container
    data = _dataset_factory(dataset_cfg)
    model = _model_factory(model_cfg)
    train(model, data)
