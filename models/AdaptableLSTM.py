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

def MRR(pred, y):
    # calculate ranks (higher % prediction is lower rank)
    ranks = -1*pred
    ranks = ranks.argsort()
    ranks = ranks.argsort()
    ranks += 1 # best rank should be 1 not 0
    # 1/rank for correct prediction entries, 0 for others (as y is 0 then)
    mrr = (y / ranks).sum(axis=1).mean()
    return mrr

def PairwiseLogLoss(pred, y):
    None

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
    
    #TODO: rework this flow to account for participants having different #s of responses
    def report_scores(self, x, y):
        with torch.no_grad():
            pred = self.forward(x).view(y.size())
            ynp = y.detach().numpy()
            prednp = pred.detach().numpy()
            pred0 = torch.tensor(np.array([i for ind, i in enumerate(prednp) if ynp[ind, 0] != 0]))
            y0 = torch.tensor(np.array([i for i in ynp if i[0] != 0]))
            pred1 = torch.tensor(np.array([i for ind, i in enumerate(prednp) if ynp[ind, 0] == 0]))
            y1 = torch.tensor(np.array([i for i in ynp if i[0] == 0]))
            print(pred1, y1)
            k = self.fc_q1.out_features
            crit = torch.nn.MSELoss()
            mseloss = crit(pred[:,:k], y[:,:k]) + crit(pred[:,-1*k:], y[:,-1*k:])
            mseloss0 = crit(pred0[:,:k], y0[:,:k]) + crit(pred0[:,-1*k:], y0[:,-1*k:])
            mseloss1 = crit(pred1[:,:k], y1[:,:k]) + crit(pred1[:,-1*k:], y1[:,-1*k:])
            crit = torch.nn.CrossEntropyLoss()
            celoss = crit(pred[:,:k], y[:,:k]) + crit(pred[:,-1*k:], y[:,-1*k:])
            celoss0 = crit(pred0[:,:k], y0[:,:k]) + crit(pred0[:,-1*k:], y0[:,-1*k:])
            celoss1 = crit(pred1[:,:k], y1[:,:k]) + crit(pred1[:,-1*k:], y1[:,-1*k:])
            crit = NDCG
            ndcg = crit(pred[:,:k], y[:,:k]) + crit(pred[:,-1*k:], y[:,-1*k:])
            ndcg0 = crit(pred0[:,:k], y0[:,:k]) + crit(pred0[:,-1*k:], y0[:,-1*k:])
            ndcg1 = crit(pred1[:,:k], y1[:,:k]) + crit(pred1[:,-1*k:], y1[:,-1*k:])
            crit = MRR
            mrr = crit(pred[:,:k], y[:,:k]) + crit(pred[:,-1*k:], y[:,-1*k:])
            mrr0 = crit(pred0[:,:k], y0[:,:k]) + crit(pred0[:,-1*k:], y0[:,-1*k:])
            mrr1 = crit(pred1[:,:k], y1[:,:k]) + crit(pred1[:,-1*k:], y1[:,-1*k:])
            return np.array([mseloss.item(), celoss.item(), ndcg.item(), mrr.item(), mseloss0.item(), celoss0.item(), ndcg0.item(), mrr0.item(), mseloss1.item(), celoss1.item(), ndcg1.item(), mrr1.item()])/2, ["MSE", "CE", "NDCG", "MRR", "MSE0", "CE0", "NDCG0", "MRR0", "MSE1", "CE1", "NDCG1", "MRR1"]
