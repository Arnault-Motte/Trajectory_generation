import torch
import torch.nn as nn


class Early_stopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float=0.0,
        best_model_path: str="data_orly/src/generation/models/saved_temp/best_model.pth",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.path = best_model_path
        self.min_loss = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, validation_loss: float, model: nn.Module) -> None:
        if validation_loss < self.min_loss - self.min_delta: #making sure it stays around the good valuue
            if validation_loss < self.min_loss:
                self.min_loss = validation_loss
                torch.save(model.state_dict(), self.path) #saving best model
            #self.min_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter > self.patience:
            self.stop = True
            print("|--Early stopping--|")
