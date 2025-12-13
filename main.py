import os
import torch.nn as nn

CONFIG = {

}

class EvolutionNet(nn.Module):
    def __init__(self):
        super(EvolutionNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(CONFIG["N_SENSORS"], 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2), # Steering, Accel
            nn.Tanh() # contraint between -1 and 1
        )
        
    def forward(self, x):
        return self.fc(x)