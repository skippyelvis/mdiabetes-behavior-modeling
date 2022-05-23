from utils.behavior_data import BehaviorData
from models.basic import BasicLSTM
import torch
import numpy as np

class Experiment:
    
    def __init__(self, data_kw={}, model_kw={}):
        self.data_kw = data_kw
        self.model_kw = model_kw
        self.bd = BehaviorData(**data_kw)
        self.model = BasicLSTM(
            input_size=self.bd.dimensions[0],
            output_size=self.bd.dimensions[1],
            **model_kw,
        )
        
    def train(self, epochs=1, **kw):
        ls = []
        for e in range(epochs):
            ls.append(self.train_epoch(**kw))
        return ls
        
    def train_epoch(self, iters=None, **kw):
        # train for one epoch (all subjects, all series)
        # visit each (subj,ser) iters times
        # return a list of loss per series per subject
        #       [[11loss,12loss,...],[21loss,22loss,...],...]
        loss_history = []
        crit = self.model.make_criterion()
        opt = self.model.make_optimizer()
        for subj in self.bd.iterate_subjects(n_subj=kw.get("n_subj")):
            subj_loss_history = []
            for x, y in self.bd.iterate_series(subj, n_ser=kw.get("n_ser")):
                l = self.train_step(crit, opt, x, y, iters)
                subj_loss_history.append(l)
            loss_history.append(subj_loss_history)
        return loss_history
                
    def train_step(self, crit, opt, x, y, iters=None):
        # train for iters steps and return mean loss value
        if iters is None:
            iters = 1
        x, y = self.totensor(x,y)
        lossh = []
        for i in range(iters):
            opt.zero_grad()
            pred = self.model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            lossh.append(loss.item())
        return np.mean(lossh)
    
    def totensor(self, *args):
        for a in args:
            a = torch.Tensor(a)
            a = a.view(a.shape[0], 1, a.shape[1])
            yield a
