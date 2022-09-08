from torch import nn
import numpy as np
import torch
from models.base import Base, N_RESP_CATEGORIES

# NOT DIFFERENTIABLE
def NDCG(pred, y):
    # calculate normalized discounted cumulative gain
    # calculate ranks (higher % prediction is lower rank)
    ranks = -1*pred
    ranks = ranks.argsort()
    ranks = ranks.argsort()
    ranks += 1 # best rank should be 1 not 0
    dcg = (torch.exp2(y) - 1) / torch.log2(1 + ranks)
    # ideal ranks (take y, set all zero terms to 3 to avoid div by 0)
    # actual rank of zero terms in y is irrelevant (2^0 - 1 = 0)
    iranks = y.clone()
    iranks[y == 0] = 3
    # calculate dcg of ideal predictions (ideal dcg)
    idcg = (torch.exp2(y) - 1) / torch.log2(1 + iranks)
    # divide dcg by idcg to get ndcg
    loss = dcg.sum() / idcg.sum()
    return loss

def ApproxNDCG(pred, y):
    # differentiable approximate form of NDCG
    None
    

class AdaptableLSTM(Base):
    
    def __init__(self, *args, **kw):
        # Simple model: lstm block with two linear output layers,
        #    one to predict the response of either label (question)
        # input_size:  size of linear input to model
        # hidden_size: size of lstm hidden layer
        # output_size: size of output label of data
        super().__init__(*args, **kw)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.fc_q1 = nn.Linear(self.hidden_size, self.output_size//2)
        self.fc_q2 = nn.Linear(self.hidden_size, self.output_size//2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # One forward pass of input vector/batch x
        # Pass x thru the LSTM cell, then pass output
        #    through each linear output layer for each
        #    response prediction. 
        # Then join predictions together as one vector
        H0, C0 = self._init_hc()
        output, (H,C) = self.lstm(x, (H0, C0)) 
        out = self.relu(output)
        out_q1 = self.fc_q1(output).softmax(2)
        out_q2 = self.fc_q2(output).softmax(2)
        return torch.cat([out_q1, out_q2],2)
    
    
    def train_step(self, x, y):
        # One optimization step of our model on 
        #    input data (x,y)
        # Returns loss value
        opt = self.make_optimizer()
        if (self.lossfn == "NDCG"):
            crit = NDCG
        else:
            crit = self.make_criterion()
        opt.zero_grad()
        pred = self.forward(x).view(y.size())
        k = self.fc_q1.out_features
        loss = crit(pred[:,:k], y[:,:k]) + crit(pred[:,-1*k:], y[:,-1*k:])
        loss.backward()
        opt.step()
        return loss.item()
    
    def report_scores(self, x, y):
        with torch.no_grad():
            pred = self.forward(x).view(y.size())
            k = self.fc_q1.out_features
            crit = torch.nn.MSELoss()
            mseloss = crit(pred[:,:k], y[:,:k]) + crit(pred[:,-1*k:], y[:,-1*k:])
            crit = torch.nn.CrossEntropyLoss()
            celoss = crit(pred[:,:k], y[:,:k]) + crit(pred[:,-1*k:], y[:,-1*k:])
            crit = NDCG
            ndcg = crit(pred[:,:k], y[:,:k]) + crit(pred[:,-1*k:], y[:,-1*k:])
            return np.array([mseloss.item(), celoss.item(), ndcg.item()]), ["MSE", "CE", "NDCG"]
