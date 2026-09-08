import torch
import numpy as np
import inspect
from tabpfn import TabPFNClassifier

class tabpfn(torch.nn.Module):
    def __init__(self, tasktype):
        
        super(tabpfn, self).__init__()
        self.tasktype = tasktype
        if tasktype == "regression":
            raise ValueError("The official TabPFN baseline supports classification only")
        if "N_ensemble_configurations" not in inspect.signature(TabPFNClassifier).parameters:
            raise RuntimeError("Official MultiTab requires the legacy TabPFN API; install tabpfn==0.1.11 in the benchmark environment")
        self.model = TabPFNClassifier(device=torch.device("cpu"), N_ensemble_configurations=32)
    
    def fit(self, X_train, y_train, X_val, y_val):
        if self.tasktype == "multiclass":
            y_train = torch.argmax(y_train, dim=1)
        self.model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
            
    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())
        
    def predict_proba(self, X_test, logit=False):
        probabilities = self.model.predict_proba(X_test.cpu().numpy())
        if not logit:
            return probabilities
        probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
        if self.tasktype == "binclass":
            return np.log(probabilities[:, 1] / (1 - probabilities[:, 1]))
        return np.log(probabilities)
