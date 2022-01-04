import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):
    '''
    version 1.1
    '''
    def __init__(self, num_classes):        
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv1d(3, 63, kernel_size = 12, groups = 3) # Depthwise Convolution
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv1d(63, 64, kernel_size = 12, stride = 2)
        self.relu1_2 = nn.ReLU()
        
        self.conv2_1 = nn.Conv1d(64, 128, kernel_size = 12)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv1d(128, 128, kernel_size = 12, stride = 2)
        self.relu2_2 = nn.ReLU()
        
        self.conv3_1 = nn.Conv1d(128, 128, kernel_size = 8)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size = 8, stride = 2)
        self.relu3_2 = nn.ReLU()
        
        self.conv4_1 = nn.Conv1d(128, 256, kernel_size = 8)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv1d(256, 256, kernel_size = 8, stride = 2)
        self.relu4_2 = nn.ReLU()
        
        self.conv5_1 = nn.Conv1d(256, 256, kernel_size = 8)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv1d(256, 256, kernel_size = 8, stride = 2)
        self.relu5_2 = nn.ReLU()
        
        self.conv6_1 = nn.Conv1d(256, 512, kernel_size = 3)
        self.relu6_1 = nn.ReLU()
        self.conv6_2 = nn.Conv1d(512, 512, kernel_size = 3, stride = 2)
        self.relu6_2 = nn.ReLU()
        
        self.conv7_1 = nn.Conv1d(512, 1024, kernel_size = 3)
        self.relu7_1 = nn.ReLU()
        self.conv7_2 = nn.Conv1d(1024, 1024, kernel_size = 3)
        self.relu7_2 = nn.ReLU()
        
        self.conv8_1 = nn.Conv1d(1024, 4096, kernel_size = 3)
        self.relu8_1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        
        x = self.conv6_1(x)
        x = self.relu6_1(x)
        x = self.conv6_2(x)
        x = self.relu6_2(x)
                
        x = self.conv7_1(x)
        x = self.relu7_1(x)
        x = self.conv7_2(x)
        x = self.relu7_2(x)
                        
        x = self.conv8_1(x)
        x = self.relu8_1(x)
        x = x.view(x.size(0), -1)
        x = self.drop1_fc1(x)
        x = self.fc1(x)
        x = x.squeeze(1)
        
        return x