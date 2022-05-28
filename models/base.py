import torch.nn as nn
from torch.autograd import Variable
import torch

# our categories
RESP_CATEGORIES = [0, 1, 2, 3]
N_RESP_CATEGORIES = len(RESP_CATEGORIES)

# base class for models, mostly just utility code, not a real model
class Base(nn.Module):
    
    def __init__(self, 
                 input_size=36, hidden_size=256, output_size=8,
                 lossfn="CrossEntropyLoss", loss_kw={},
                 optimizer="SGD", opt_kw={"lr": 1e-3},):
        # define all inputs to the model
        # input_size:   # features in input
        # hidden_size:  # size of hidden layer
        # output_size   # features in output
        # lossfn:       # string representing torch loss fn
        # loss_kw:      # keyword arguments to loss fn
        # optimizer:    # string represneting torch optimizer
        # opt_kw:       # keyword arguments to optimizer
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lossfn, self.loss_kw = lossfn, loss_kw
        self.optimizer, self.opt_kw = optimizer, opt_kw
        
    def forward(self, x):
        # fake forward function so we can do other stuff
        return x
    
    def predict(self, x):
        # call forward method but do not collect gradient
        with torch.no_grad():
            return self.forward(x)
        
    def make_criterion(self):
        # build the loss function
        return getattr(torch.nn, self.lossfn)(**self.loss_kw)
    
    def make_optimizer(self):
        # build the optimizer instance
        optcls = getattr(torch.optim, self.optimizer)
        opt = optcls(self.parameters(), **self.opt_kw)
        return opt
        
    def _init_hc(self):
        # initialize the hidden/cell state variables
        h0 = Variable(torch.zeros(1, 1, self.hidden_size))
        c0 = Variable(torch.zeros(1, 1, self.hidden_size))
        return h0, c0
                     
                     
            