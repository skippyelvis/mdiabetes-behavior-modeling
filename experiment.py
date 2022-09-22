from utils.behavior_data import BehaviorData
import torch
import numpy as np
import importlib

class Experiment:
    
    def __init__(self, data_kw={}, model="BasicLSTM", model_kw={}, train_kw={}):
        # data_kw:  dict of keyword arguments to BehaviorData instance
        # model_kw: dict of keyword arguments for Model instance
        # train_kw: dict of keyword arguments for training loop
        self.data_kw = data_kw
        self.model_name = model
        self.model_kw = model_kw
        self.train_kw = train_kw
        self.bd = BehaviorData(**data_kw)
        self.model = self._get_model()(
            input_size=self.bd.dimensions[0],
            output_size=self.bd.dimensions[1],
            **model_kw,
        )
        
    def run(self):
        # Train the model, report parameters and results
        rep = self.__dict__
        rep = {k: v for k,v in rep.items() if "_kw" in k}
        train_loss = self.train()
        results = self.evaluate()
        rep["params"] = {
            "data_kw": self.data_kw,
            "model_name": self.model_name,
            "model_kw": self.model_kw,
            "train_kw": self.train_kw,
        }
        rep["results"] = results
        rep["loss"] = train_loss
        return rep
        
    def train(self):
        # Loop over data and train model on each batch
        # Returns matrix of loss for each participant
        out = []
        epochs = self.train_kw.get("epochs", 1)
        rec_every = self.train_kw.get("rec_every", 5)
        for e in range(epochs):
            lh = self.train_epoch()
            if (e%rec_every) == 0:
                out.append(lh)
        return np.array(out)
    
    def train_epoch(self):
        # feed through training data one time
        x, y = self.bd.features[self.bd.train].unsqueeze(1), self.bd.labels[self.bd.train]
        # print(x.shape, y.shape)
        loss = self.model.train_step(x, y)
        return [loss]
    
    def evaluate(self):
        # Evaluate the trained models predictions
        n_subj = self.train_kw.get("n_subj")
        evals = []
        x, y = self.bd.features[self.bd.test].unsqueeze(1), self.bd.labels[self.bd.test]
        pred = self.model.predict(x).view(y.shape)
        pred = pred.view(y.shape)
        evals.append(self.diff_matrix(y, pred))
        return evals
    
    def report_scores(self):
        x, y = self.bd.features[self.bd.test].unsqueeze(1), self.bd.labels[self.bd.test]
        res, label = self.model.report_scores(x, y)
        return res, label
        
            
    def diff_matrix(self, true, pred):
        # Build a matrix showing the error in predicted responses
        diff = np.zeros((2, true.shape[0]))
        diff[0] = (true[:,:4].argmax(1)-pred[:,:4].argmax(1))
        diff[1] = (true[:,-4:].argmax(1)-pred[:,-4:].argmax(1))
        return diff
        
    def totensor(self, a):
        # Convert a to a batched tensor
        a = torch.Tensor(a)
        a = a.view(a.shape[0], 1, a.shape[1])
        return a
    
    def forceBatch(self, a):
        # Convert a to a batched tensor
        a = a.view(1, 1, a.shape[0])
        return a
        
    def _get_model(self):
        mod = importlib.import_module(f"models.{self.model_name}")
        return getattr(mod, self.model_name)
