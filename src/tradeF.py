# 必要なモジュールのインポート
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

#　ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(273, 200)
        self.fc2 = nn.Linear(200, 2)
        
    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        return h