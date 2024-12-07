from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, input_tensor, window_steps=30, label_steps=1):
        self.input_tensor = input_tensor
        self.window_steps = window_steps
        self.label_step = label_steps

    def __len__(self):
        return self.input_tensor.shape[0] - self.window_steps - self.label_step

    def __getitem__(self, idx):
        return self.input_tensor[idx : idx + self.window_steps], self.input_tensor[
            idx + self.window_steps : idx + self.window_steps + self.label_step
        ]
