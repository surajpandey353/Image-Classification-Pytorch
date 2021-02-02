import torch
import torch.nn as nn


class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #############################
        # Initialize your network
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size = 3,stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.cnn_layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.cnn_layer4 =  nn.Sequential(           
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*8*8, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1000, 300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(300, 100),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(100, 8)
             
      )

        #############################
        
    def forward(self, x):
        
        #############################
        # Implement the forward pass
        x1 = self.cnn_layer1(x)
        x2 = self.cnn_layer2(x1)
        x3 = x2 + x1
        x4 = self.cnn_layer3(x3)
        x5 = self.cnn_layer4(x4)
        #x = x.view(x.size(0), -1)
        x6 = self.linear_layers(x5)
        return x6
        #############################
        
        pass
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model')

