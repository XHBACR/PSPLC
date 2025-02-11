
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, train_set): #(表格全部信息、映射id、时间分段情况)
        self.trajectories=train_set
        print(f"Dataset length: ", len(self.trajectories))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, item):
        return self.trajectories[item]

