'''
Created on Oct 22, 2026
Artificial Neural Network (ANN) using linear combiners, ReLU activation functions, and Cross Entropy loss function.
Otherwise known as a multi-layer perceptron (MLP) using back-propagation of errors to train the weights and biases.
The task is to classify four categories (classes), each having ten properties uniformly distributed in regions of (0,256].
Display the learning curve and the distribution of the properties for each class.  Test the trained ANN with a separate
set of samples and tabulate the results.

The PyTorch framework was used to build the ANN.

@author: Tom Teplick
'''
import torch
from torch import nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

nclasses = 4
nproperties = 10

# hidden layer width
layerWidth = 15

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
    
    def train_(self, dataloader, loss_fn, optimizer, device):
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            # Compute prediction error
            pred = self.forward(X)
            loss = loss_fn(pred, y.squeeze(1))
    
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # save the loss for learning curve display
            self.loss[batch] = loss.item()
            
    def test(self, dataloader, device):
        self.eval()
        with torch.no_grad():
            for X, y in dataloader:
                #print(f"outer={outer}, X={X}, y={y}")
                X, y = X.to(device), y.to(device)
                pred = self.forward(X)
                #print(f"\npred={pred}")
                #print(f"\ny={y}, size={y.size()}")
                # loop over y tensor and determine if pred[i].argmax() is equal,
                # count the number of samples for each class
                for i,target in enumerate(y):
                    #print(f"i={i}, target={target}")
                    self.count[target.item()] += 1
                    if pred[i].argmax() == target :
                        self.correct[target.item()] += 1
                        
    def displayTestResults(self):
               
        # show classes, count, correct, %correct, totals
        # heading
        ttlCorrect = 0
        ttlCount = 0
        sep = 8
        horzBorder = '|' + '-'*sep + '|' + '-'*sep + '|' + '-'*sep + '|' + '-'*sep + '|'
        print(horzBorder)
        print("|{0:8s}|{1:8s}|{2:8s}|{3:8s}|".format("Class", "Count", "Correct", "%Correct"))
        print(horzBorder)
        for i, val in enumerate(self.count):
            print("|{0:<8d}|{1:<8d}|{2:<8d}|{3:<8d}|".format(i+1, val, self.correct[i], self.correct[i]*100//val))
            print(horzBorder)
            ttlCount += val
            ttlCorrect += self.correct[i]
        print("|{0:<8s}|{1:<8d}|{2:<8d}|{3:<8d}|".format("Totals", ttlCount, ttlCorrect, ttlCorrect*100//ttlCount))
        print(horzBorder)
        
    def  plotClassDistribution(self, propbegin, propend, maxProp):
        # plot the ranges of the class properties which are regions in (0, 255)
        # x-axis is a unitless number running from 0 to 10*prop-space + 4*(class-space+class-width) + property-space
        # y-axis is an integer running from 0 to 255
        # loop over 10 properties
        #   skip property space
        #   loop over 4 property ranges with space in-between each property range
        # set x-label, y-label, title, xticks, classSpace, propSpace, legend, line colors, line width
        # show figure
        
        # x-axis spacing between items: classes and properties
        classSpace = 2
        propSpace = nclasses*classSpace
        # line colors
        clr = np.array(['red', 'green', 'blue', 'orange'])
        # line properties: width, style
        lnwid = 3
        lnsty = "solid"
        lbl = np.array(['class1', 'class2', 'class3', 'class4'])
        # x,y coordinates arrays to be scaled and sliced
        xcoord = np.ones(maxProp, dtype=np.uint8)
        ycoord = np.arange(maxProp, dtype=np.uint8)
        xtick = np.zeros(nproperties, dtype=np.uint8)
                
        x = 0
        _, ax = plt.subplots(1,1)
        # loop over the properties
        for prop in range(nproperties):
            x += propSpace
            # loop over the classes
            for cls in range(nclasses):
                # property y length
                ln = propend[cls][prop] - propbegin[cls][prop]
                if prop == 0:
                    ax.plot(x*xcoord[:ln], ycoord[propbegin[cls][prop]:propend[cls][prop]], label=lbl[cls],
                            color=clr[cls], linewidth=lnwid, linestyle=lnsty)
                else:
                    ax.plot(x*xcoord[:ln], ycoord[propbegin[cls][prop]:propend[cls][prop]],
                            color=clr[cls], linewidth=lnwid, linestyle=lnsty)                    
                # put the property value at this x coordinate
                if cls == nclasses//2:
                    xtick[prop] = x
                x += classSpace
        ax.legend()
        ax.set_xlabel('Class Properties')
        ax.set_ylabel('Property Values')
        ax.set_title('Property Distribution')
        # custom x-ticks for properties
        ax.set_xticks(xtick, np.arange(nproperties)+1)
        plt.show(block=True)  
                
    def plotLearningCurve(self):
        # show cross entropy loss per epoch from training the MLP
        _, ax = plt.subplots(1,1)
        x = np.arange(self.loss.size(0), dtype=np.uint32)
        y = self.loss.numpy()
        # line properties: width, style
        lnwid = 3
        lnsty = "solid"
        ax.plot(x, y, linewidth=lnwid, linestyle=lnsty)        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross Entropy Loss')
        ax.set_title('Learning Curve')
        plt.show(block=True)          
                                          
def runLinearReluCrossEntropy():
    
    parser = argparse.ArgumentParser(prog='linear_relu_crossentry', description="MLP neural network (ReLU, Cross Entropy Loss) for classification using PyTorch")
    parser.add_argument('epochs', type=int, choices=[10, 20, 50, 100, 200, 500, 1000])
    parser.add_argument('batch_size', type=int, choices=[1, 2, 10, 32, 64])
    parser.add_argument('learning_rate', type=float, choices=[.001, .002, .01, .02])
    parser.add_argument('momentum', type=float, choices=[.5, .6, .7, .8, .9])
    args = parser.parse_args()
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    # seed the random number generator
    torch.manual_seed(int(time.time()))
    
    # each epoch consists of batch_size samples which consists of (input,target)
    epochs = args.epochs  #args.epochs
    # number of samples consisting of (input,target)
    batch_size = args.batch_size #args.batch_size
    # learning rate to scale the backpropagation error
    learning_rate = args.learning_rate #args.learning_rate
    # momentum
    momentum = args.momentum

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
    print(f"Model: {model}")    
    
    # Train the model
    
    # create samples from the begin/end properties            
    model.createSamples(nsamples, input_data, target, propbegin, propend)
        
    # Create dataset from sample tensors, normalize the input data to (0,1]
    dataset = data.TensorDataset(input_data/maxProp, target)
    # Create dataloader from dataset
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Display the class distributions using matplotlib
    print("Display the class distributions")
    model.plotClassDistribution(propbegin, propend, maxProp)
       
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
            
    print("Begin training the model.")
    for t in range(epochs):
        print("*", end='')
        model.train_(dataloader, loss_fn, optimizer, device)
        if (t+1) % 50 == 0:
            print()
    print("\nDone training")

    # Display Cross Entropy Loss using matplotlib
    print("Display the Cross Entropy Loss")
    model.plotLearningCurve()
    
    torch.save(model.state_dict(), linearReLUcrossEntropyModel)
    print("Saved PyTorch Model State to %s" % linearReLUcrossEntropyModel)
    
    # Save the class properties
    with open(linearReLUcrossEntropyData, mode='w', encoding="utf-8") as fout:
        for cls in range(nclasses):
            for prop in range(nproperties):
                print(f"{propbegin[cls][prop]} {propend[cls][prop]} ", file=fout, end='')
            print(file=fout)
            
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
        model.test(dataloader, device)
        if (t+1) % 50 == 0:
            print()
    print("\nDone testing.  Test results:")
        
    # Tabulate and display the test results: class, count, correct, %correct, totals
    model.displayTestResults()
    
if __name__ == '__main__':
    runLinearReluCrossEntropy()