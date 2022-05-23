from torch import nn
from models.base import Base, N_RESP_CATEGORIES

class BasicLSTM(Base):
    
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h0, c0 = self._init_hc(x)
        output, (hn, cn) = self.lstm(x, (h0, c0)) 
        out = self.relu(output)
        out = self.fc1(out)
        return out