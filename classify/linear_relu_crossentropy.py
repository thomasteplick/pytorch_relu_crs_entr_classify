'''
Created on Oct 22, 2026
Artificial Neural Network (ANN) using linear combiners, ReLU activation functions, and Cross Entropy loss function
Classify four categories each having ten properties uniformly distributed in regions of (0,256]

@author: Tom Teplick
'''
import torch
from torch import nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import argparse
from threading import Thread

nclasses = 4
nproperties = 10

# hidden layer width
layerWidth = 15
# learning rate
learnintRate = .001

# momentum
momentum = 0.9

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(nproperties, layerWidth),
            nn.ReLU(),
            nn.Linear(layerWidth, layerWidth),
            nn.ReLU(),
            nn.Linear(layerWidth, nclasses)
        )
        self.loss = torch.zeros(epochs, dtype=torch.float32)
        self.count = torch.zeros(nclasses, dtype=torch.int32)
        self.correct = torch.zeros(nclasses, dtype=torch.int32)
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    def createSamples(self, nsamples, input_data, target, propbegin, propend):
        # Create the input data and the target
        for samp in range(nsamples):
            for prop in range(nproperties):
                cls = torch.randint(0, nclasses, (1,1), dtype=torch.uint8)
                target[samp] = cls.item()
                input_data[samp][prop] = torch.randint(propbegin[cls.item()][prop], 
                    propend[cls.item()][prop], (1,1), dtype=torch.uint8).item()
    
    def train(self, dataloader, loss_fn, optimizer):
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            # Compute prediction error
            pred = self.forward(X)
            loss = loss_fn(pred, y)
    
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # save the loss for learning curve display
            self.loss[batch] = loss.item()
            
    def test(self, dataloader):
        self.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.forward(X)
                # loop over y tensor and determine if pred[i].argmax() is equal,
                # count the number of samples for each class
                for i,target in enumerate(y):
                    self.count[target] += 1
                    if pred[i].argmax() == target :
                        self.correct[target] += 1
                        
    def displayTestResults(self):
               
        # show classes, count, correct, %correct, totals
        # heading
        ttlCorrect, ttlCount = 0,0
        print("{0:8s}{1:8s}{2:8s}{3:8s}".format("Class", "Count", "Correct", "%"))
        print('-' * 32)
        for i, val in enumerate(self.count):
            print("{0:<8d}{1:<8d}{2:<8d}{3:8d}".format(i+1, val, self.correct[i], self.correct[i]*100//val))
            ttlCount += val
            ttlCorrect += self.correct
        print('=' * 32)
        print("{0:8s}{1:8d}{2:8d}{3:8d}".format("Totals", ttlCount, ttlCorrect, ttlCorrect*100//ttlCount))
        
    def  plotClassDistribution(self):
        # plot the ranges of the class properties which are regions in (0, 255)
        # x-axis is a unitless number running from 0 to 10*prop-space + 4*(class-space+class+class-width) + property-space
        # y-axis is an integer running from 0 to 255
        # loop over 10 properties
        #   skip property space
        #   loop over 4 property ranges with space in-between each property range
        # skip property space
        # set x-labels, y-labels, x-axis, y-axis, title, xticks, classSpace, propSpace
        pass
                              
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
            
def runLinearReluCrossEntropy():
    
    parser = argparse.ArgumentParser(prog='linear_relu_crossentry', description="neural network for classification using PyTorch")
    parser.add_argument('epochs', type=int, choices=[0,1,2,3])
    parser.add_argument('batch_size', type=int, choices=[0,1,2])
    parser.add_argument('learning_rate', type=float, choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
    args = parser.parse_args()

    epochs = 20
    batch_size = 1
    maxProp = 256
    maxBegin = maxProp//2
    nsamples = epochs*batch_size
    linearReLUcrossEntropyModel = "data\\linearReLUcrossEntropy.pth"
    linearReLUcrossEntropyData = "data\\LinearReLUcrossEntropy.txt"
   
    # Set class properties bounds tensors
    propbegin = torch.randint(0, maxBegin, (nclasses, nproperties), dtype = torch.uint8)   
    propend=torch.randint(1, maxBegin, (nclasses, nproperties), dtype=torch.uint8) + propbegin
    
    # Input data and the target tensors comprise the samples used to train the ANN
    input_data = torch.empty(nsamples, nproperties, dtype=torch.uint8)
    target = torch.empty(nsamples, 1, dtype=torch.uint8)
    
    # create the ANN
    model = NeuralNetwork(epochs).to(device)
    print(model)    
    
    # create samples from the begin/end properties            
    model.createSamples(nsamples, input_data, target, propbegin, propend)            
        
    # Create dataset from sample tensors, normalize the input data to (0,1]
    dataset = data.TensorDataset(input_data/maxProp, target)
    # Create dataloader from dataset
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Display the class distributions in a thread
    print("Display the class distributions in a separate window")
    t = Thread(target=model.plotClassDistribution)
    t.run()
       
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learnintRate, momentum=momentum)
            
    print("Begin training with the model.")
    for t in range(epochs):
        print("*", end='')
        model.train(dataloader, loss_fn, optimizer)
        if (t+1) % 50 == 0:
            print("\n")
    print("Done training")

    # Display Cross Entropy Loss using matplotlib in a thread
    print("Display the Cross Entropy in a separate window")
    
    torch.save(model.state_dict(), linearReLUcrossEntropyModel)
    print("Saved PyTorch Model State to %s" % linearReLUcrossEntropyModel)
    
    # Save the class properties
    with open(linearReLUcrossEntropyData, mode='w', encoding="utf-8") as fout:
        for cls in range(nclasses):
            for prop in range(nproperties):
                print(f"{propbegin[cls][prop]} {propend[cls][prop]} ", file=fout, end='')
            print('\n', file=fout)
        
    # Test the model using different samples
    
    # create samples from the begin/end properties            
    model.createSamples(nsamples, input_data, target, propbegin, propend)            
    # Create dataset from sample tensors, normalize the input data to (0,1]
    dataset = data.TensorDataset(input_data/maxProp, target)
    # Create dataloader from dataset
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Begin testing the model")
    for t in range(epochs):
        print("*", end='')
        model.test(dataloader)
        if (t+1) % 50 == 0:
            print("\n")
    print("Done testing")
        
    # Tabulate and display the test results: class, count, correct, %correct, totals
    model.displayTestResults()
    
if __name__ == '__main__':
    runLinearReluCrossEntropy()