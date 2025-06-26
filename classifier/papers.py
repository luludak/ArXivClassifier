import torch
from torch.utils.data import Dataset

class PapersDataset(Dataset):
    def __init__(self, X_sparse, Y):
        self.X = X_sparse
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        # Define tensor mapping.
        x_row = self.X[idx].toarray().squeeze()
        x_tensor = torch.tensor(x_row, dtype=torch.float32)
        return x_tensor, self.Y[idx]