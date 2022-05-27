from torch import nn
import torch
from models.base import Base, N_RESP_CATEGORIES

class BasicLSTM(Base):
    
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.fc_q1 = nn.Linear(self.hidden_size, self.output_size//2)
        self.fc_q2 = nn.Linear(self.hidden_size, self.output_size//2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        H0, C0 = self._init_hc()
        output, (H,C) = self.lstm(x, (H0, C0)) 
        out = self.relu(output)
        out_q1 = self.fc_q1(output).softmax(2)
        out_q2 = self.fc_q2(output).softmax(2)
        return torch.cat([out_q1, out_q2],2)
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
    
    def train_step(self, x, y):
        opt = self.make_optimizer()
        crit = self.make_criterion()
        opt.zero_grad()
        pred = self.forward(x).view(y.size())
        loss = crit(pred[:,:4], y[:,:4]) + crit(pred[:,-4:], y[:,-4:])
        loss.backward()
        opt.step()
        return loss.item()