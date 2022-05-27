from utils.behavior_data import BehaviorData
from models.basic import BasicLSTM
import torch
import numpy as np

class Experiment:
    
    def __init__(self, data_kw={}, model_kw={}, train_kw={}):
        self.data_kw = data_kw
        self.model_kw = model_kw
        self.train_kw = train_kw
        self.bd = BehaviorData(**data_kw)
        self.model = BasicLSTM(
            input_size=self.bd.dimensions[0],
            output_size=self.bd.dimensions[1],
            **model_kw,
        )
        
    def run(self):
        rep = self.__dict__
        rep = {k: v for k,v in rep.items() if "_kw" in k}
        train_loss = self.train()
        results = self.evaluate()
        rep["results"] = results
        rep["loss"] = train_loss
        return rep
        
    def train(self):
        out = []
        epochs = self.train_kw.get("epochs", 1)
        rec_every = self.train_kw.get("rec_every", 5)
        for e in range(epochs):
            lh = self.train_epoch()
            if (e%rec_every) == 0:
                out.append(lh)
        return np.array(out)
    
    def train_epoch(self):
        n_subj = self.train_kw.get("n_subj", None)
        loss_history = []
        for (x, y) in self.bd.iterate(n_subj):
            x, y = self.totensor(x), torch.Tensor(y)
            loss = self.model.train_step(x, y)
            loss_history.append(loss)
        return loss_history
    
    def evaluate(self):
        n_subj = self.train_kw.get("n_subj")
        evals = []
        for (x, y) in self.bd.iterate(n_subj):
            x, y = self.totensor(x), torch.Tensor(y)
            pred = self.model.predict(x).view(y.shape)
            pred = pred.view(y.shape)
            evals.append(self.diff_matrix(y, pred))
        return evals
            
    def diff_matrix(self, true, pred):
        diff = np.zeros((2, true.shape[0]))
        diff[0] = (true[:,:4].argmax(1)-pred[:,:4].argmax(1))
        diff[1] = (true[:,-4:].argmax(1)-pred[:,-4:].argmax(1))
        return diff
        
    def totensor(self, a):
        a = torch.Tensor(a)
        a = a.view(a.shape[0], 1, a.shape[1])
        return a
