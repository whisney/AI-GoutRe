import torch
import numpy as np
from torch import nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import copy

class MLP(nn.Module):
    def __init__(self, in_features_num, num_class, last_activate=False):
        super().__init__()
        self.last_activate = last_activate
        self.net = nn.Sequential(
            nn.Linear(in_features_num, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.LayerNorm(50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, num_class),
        )

    def forward(self, x):
        if self.last_activate:
            return torch.softmax(self.net(x), dim=1)
        else:
            return self.net(x)

class My_Dataset(Dataset):
    def __init__(self, X_data, Y_label, augment=True):
        self.X_data = X_data
        self.Y_label = Y_label
        self.augment = augment
        self.len = self.X_data.shape[0]

    def __getitem__(self, idx):
        X = self.X_data[idx]
        Y = self.Y_label[idx]
        if self.augment:
            if np.random.rand() < 0.95:
                X += np.random.randn(*X.shape) * 0.05
        X = torch.from_numpy(X)
        Y = torch.tensor(Y)
        return X, Y

    def __len__(self):
        return self.len

def train_ANN(train_input, train_label):
    all_epoch = 500
    lr = 0.00005
    bs = 512
    in_features_num = train_input.shape[1]
    num_class = 2

    X_train, X_val, y_train, y_val = train_test_split(train_input, train_label, test_size=0.2, stratify=train_label)

    train_data = My_Dataset(X_train, y_train, augment=True)
    val_data = My_Dataset(X_val, y_val, augment=False)
    train_dataloader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MLP(in_features_num, num_class).to(device)

    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.0001, momentum=0.99, nesterov=True)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    best_AUC = 0
    best_model = None

    for epoch in range(all_epoch):
        net.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print('[%d/%d, %d/%d]' % (epoch + 1, all_epoch, i + 1, len(train_dataloader)))

        with torch.no_grad():
            val_epoch_labels = []
            val_epoch_scores = []
            net.eval()
            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs = inputs.to(device).float()
                outputs = net(inputs)
                pred = torch.softmax(outputs, dim=1).detach().cpu().numpy()[:, 1]
                val_epoch_labels.append(labels.numpy())
                val_epoch_scores.append(pred)
            val_epoch_labels = np.concatenate(val_epoch_labels)
            val_epoch_scores = np.concatenate(val_epoch_scores)
            AUC = roc_auc_score(val_epoch_labels, val_epoch_scores)
            if AUC > best_AUC:
                best_AUC = AUC
                # best_model = torch.clone(net).detach()
                best_model = copy.deepcopy(net)
    return best_model, device