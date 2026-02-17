import torch

class IllStopper:
    def __init__(self, patience:int=100):
        self._patinece = patience
        self._best_loss = float('inf')
        self._counter = 0
        self._best_state = None

    def step(self, val_loss:float, model:torch.nn.Module):
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            self._counter = 0
            self._best_state = {
                k:v.cpu().clone() for k, v in model.state_dict().items()
            }
            return False
        self._counter += 1
        return self._counter > self._patinece
    def best_state(self):
        return self._best_state
    
class EarlyStopper:
    def __init__(self, patience:int=10, min_delta:float=0.0):
        self._patience = patience
        self._min_delta = min_delta
        self._best_loss = float('inf')
        self._counter = 0
        self._best_state = None

    def step(self, val_loss:float, model:torch.nn.Module):
        if val_loss < self._best_loss - self._min_delta:
            self._best_loss = val_loss
            self._counter = 0
            self._best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            return False
        else:
            self._counter += 1
            return self._counter > self._patience

    def best_state(self):
        return self._best_state