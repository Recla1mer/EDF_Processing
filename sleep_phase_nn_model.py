# IMPORTS 
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# LOAD DATA
class EDF_Dataset(Dataset):
    def __init__(self):
        # load data
        # transform data
        pass

    def __len__(self):
        # return number of samples
        pass

    def __getitem__(self, idx):
        # return sample
        pass

# CREATE NETWORK
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# check if model runs and returns correct shape on random data
# model = NeuralNetwork(28*28, 64, 10)
# x = torch.rand(5, 28, 28) # mini-batch size 1, 28x28 pixels
# print(model(x).shape) # we want num classes (10) values for each image (5) -> [5, 10]

# SET DEVICE
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# HYPERPARAMETERS
learning_rate = 1e-3
batch_size = 64
epochs = 5

input_size = 28*28
num_classes = 10

# INITIALIZE NETWORK
model = NeuralNetwork(input_size, num_classes).to(device)

# Initialize the loss and optimizer functions
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

edf_dataset = EDF_Dataset()
edf_dataloader = DataLoader(edf_dataset, batch_size=64)

# TRAINING LOOP
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step() # updates the model parameters based on the gradients computed during the backward pass
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# TESTING LOOP
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


"""
CoPilot:
"""

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

# class SleepStageNet(nn.Module):
#     def __init__(self):
#         super(SleepStageNet, self).__init__()
#         self.fc1 = nn.Linear(2, 64)  # Assuming 2 features: heart rate and wrist acceleration
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, 5)  # Assuming 5 sleep stages as output

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)  # No activation here, assuming use of CrossEntropyLoss which includes Softmax
#         return x

# # Assuming `features` and `labels` are your data tensors
# # features.shape = [n_samples, 2], labels.shape = [n_samples]
# dataset = TensorDataset(features, labels)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model = SleepStageNet()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# for epoch in range(num_epochs):
#     for inputs, targets in dataloader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# # Prediction and 10-hour recheck logic to be implemented as needed
