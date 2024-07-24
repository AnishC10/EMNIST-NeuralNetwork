import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

training_data = datasets.EMNIST(root = "data", train = True, download = True, transform = ToTensor(), split = "mnist")
testing_data = datasets.EMNIST(root = "data",  train = False, download = True, transform = ToTensor(), split = "mnist")

torch.manual_seed(42)
BATCH = 256

train_dataloader = DataLoader(training_data, batch_size =BATCH, shuffle = True)
test_dataloader = DataLoader(testing_data, batch_size =BATCH, shuffle = True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__ (self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.globalp = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Sequential(nn.Flatten(), nn.Linear(16, len(datasets.EMNIST.classes)))
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x =self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.globalp(x)
        x = self.classify(x)
        return x

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)

#train test loop
epochs = 20
for epoch in range (epochs):
    for batch,(X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred =model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for X1, y1 in test_dataloader:
            X_test, y_test = X1.to(device), y1.to(device)
            test_pred = model(X_test)
            test_pred = test_pred.to(device)
            testing_loss = loss_fn(test_pred, y_test)
    accuracy = accuracy_score(y_true=y_test.cpu(), y_pred=test_pred.argmax(dim=1).cpu())
    training_acc = accuracy_score(y_true=y.cpu(), y_pred=y_pred.argmax(dim=1).cpu())
    f1 = f1_score(y_true=y_test.cpu(), y_pred=test_pred.argmax(dim=1).cpu(), average='micro')
    print(f'Epoch: {epoch}  Training Loss {loss}  Testing Loss: {testing_loss}  Training Accuracy:{training_acc} Testing Accuracy:{accuracy}  Testing f1:{f1}')
