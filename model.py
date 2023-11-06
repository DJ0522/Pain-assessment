import torch
from torch.utils.data import Dataset, DataLoader

from layer import *

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class MyDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        subject_data = {}
        for data in self.dataset:
            subject_id = data['subject_id']
            if subject_id not in subject_data:
                subject_data[subject_id] = []
            subject_data[subject_id].append(data)

        for subject_id, data_list in subject_data.items():
            num_batches = len(data_list) // self.batch_size
            for i in range(num_batches):
                batch_data = data_list[i*self.batch_size : (i+1)*self.batch_size]
                yield subject_id, batch_data

    def __len__(self):
        return len(self.dataset)

class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels = 1, out_channels = 3,kernel_size = 3, stride = 1, padding = 1),
            torch.nn.BatchNorm1d(3),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(in_channels = 3, out_channels = 6, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.BatchNorm1d(6),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.AvgPool1d(kernel_size=2, stride=2))

        ### Specify CORAL layer
        self.fc = CoralLayer(size_in=4224, num_classes=num_classes)
        ###--------------------------------------------------------------------###

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten

        ##### Use CORAL layer #####
        logits =  self.fc(x)
        probas = torch.sigmoid(logits)
        ###--------------------------------------------------------------------###

        return logits, probas