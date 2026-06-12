import torch
import torchvision
import torch.nn as nn 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, negative_sampling
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, to_hetero
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from sklearn.metrics import roc_auc_score

from torchvision import datasets, transforms 
import torchvision.models as models
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import random
#%pip install matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

#%pip install seaborn
import seaborn as sn
from torchvision.utils import make_grid

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='/Users/rustgi/Desktop/WM811k_Dataset/', transform=transform)

len(dataset)

for i in dataset.classes: 
    print(i) 

dataset[0]

for i in range(5):
    print(dataset[i])

dataset.classes

for i in range(10):
    image, label_index = dataset[i] # unpacking tuple
    image = image.permute(1, 2, 0)
    # pytorch stores images as [C, H, W], but matplotlib expects [H, W, C] 
    # reorder dimensions - take 1 first, 2, then 0 
    
    plt.imshow(image)
    plt.show()

train_frac = 0.7
val_frac = 0.15
test_frac = 0.15 

#random.shuffle(dataset)  # shuffle all chunks of data 

# split into training, validation, testing data: 

num_total = len(dataset)
num_train = int(train_frac * num_total)
num_val = int(val_frac * num_total)
num_test  = num_total - num_train - num_val   

train_data, val_data, test_data = torch.utils.data.random_split(dataset, [num_train,
                                                                          num_val,
                                                                          num_test])

# create DataLoader
train_loader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers = 2)
test_loader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers = 2)


# creating CNN model 

# convolution, padding, maxpooling help our neural network learn the features from images 

# convolution = mathematical operation which can be performed on two functions to generate third function that 
# shows how shape of one modified by other

class CNN(torch.nn.Module):
    def __init__(self, in_channels, num_classes): 
        # init method -- constructor method that initializes the model's layers and parameters 
        super(CNN, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 10 output features (num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(9216, num_classes)
        # widen the network to potentiall help it learn richer feature representations

        
    def forward(self, x):

        x = self.conv1(x)  # Apply first convolution and ReLU activation

        x = self.bn1(x)
        
        x = F.relu(x)  # Apply second convolution and ReLU activation

#        x = F.dropout(x, p=0.5, training=self.training)

        x = self.pool(x)           # Apply max pooling

        x = self.conv2(x)  # Apply first convolution and ReLU activation

        x = self.bn2(x)
        
        x = F.relu(x)  # Apply second convolution and ReLU activation

#        x = F.dropout(x, p=0.5, training=self.training)
 
        x = self.pool(x)           # Apply max pooling

        x = self.conv3(x)  # Apply first convolution and ReLU activation

        x = self.bn3(x)
        
        x = F.relu(x)  # Apply second convolution and ReLU activation

#        x = F.dropout(x, p=0.5, training=self.training)

        x = self.pool(x)           # Apply max pooling
        
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        
        x = self.fc1(x)            # Apply fully connected layer
        
        return x 

model = CNN(in_channels = 3, num_classes = 9)
#print(model)
#print(f"Flattened size: {x.shape[1]}")

# custom cnns much less stable than pretrained resnet
# and starts from random weights as opposed to resnet, which already knows 
# edges, textures, patterns 
# try 20-30 epochs 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)

for epoch in range(1, 150): 
    avg_train_loss = 0
    model.train() 
    
    for images, labels in train_loader:  
        images = images.to(device) 
        labels = labels.to(device) 

        optimizer.zero_grad() 
        outputs = model(images) 
        train_loss = F.cross_entropy(outputs, labels) 
        train_loss.backward() 
        optimizer.step() 
        
        avg_train_loss += train_loss.item() 

    avg_train_loss = avg_train_loss / len(train_loader)


    avg_test_loss = 0
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad(): 
        for images, labels in test_loader: # during testing, weights remain fixed
 #           avg_test_loss = 0
        # to provide an unbiased evaluation of the model's current state 
        # no optimizer step needs to be defined
        # also no backpropagation 
        
            images = images.to(device)
            labels = labels.to(device) 

            outputs = model(images) 
            test_loss = F.cross_entropy(outputs, labels) 
            avg_test_loss += test_loss.item() 

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

#            print(preds[:20])
#            print(labels[:20])
#            break

    accuracy = correct / total

    avg_test_loss = avg_test_loss / len(test_loader) 

    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, Test Accuracy = {accuracy:.4f}")

model.eval()

y_pred = []
y_true = []

# iterate over test data

with torch.no_grad():
    
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs) # feed Network

        # 1. Get the predicted class indices
        preds = torch.argmax(output, dim=1)
        
        # 2. Convert tensors to CPU numpy arrays and extend the lists
        y_pred.extend(preds.cpu().numpy()) 
        y_true.extend(labels.cpu().numpy())

classes = dataset.classes

# build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
    
row_sums = cf_matrix.sum(axis=1, keepdims=True)
df_cm = pd.DataFrame(cf_matrix / (row_sums + 1e-8),
                     index=classes,
                     columns=classes)

plt.figure(figsize = (8,4))
sn.heatmap(df_cm, annot=True)

#plt.xlabel('Predicted')
#plt.ylabel('Actual')
