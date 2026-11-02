<pre>
C:\Users\Thoma\eclipse-workspace\PyTorchClassify\classify>linear_relu_crossentropy.py -h
usage: linear_relu_crossentry [-h] {100,200,500,1000,2000,5000,10000} {1,2,10,16,32,64} {0.001,0.002,0.01,0.02} {0.5,0.6,0.7,0.8,0.9} {1,2,3} {10,15,20}

MLP neural network (ReLU, Cross Entropy Loss) for classification using PyTorch

positional arguments:
  {100,200,500,1000,2000,5000,10000}
                        the number of epochs for training and testing
  {1,2,10,16,32,64}     the batch size for training and testing
  {0.001,0.002,0.01,0.02}
                        learning rate hyperparameter for training
  {0.5,0.6,0.7,0.8,0.9}
                        momentum hyperparameter for training
  {1,2,3}               number of hidden layers, depth of MLP
  {10,15,20}            hidden layer width

options:
  -h, --help            show this help message and exit  
</pre>
<pre>
C:\Users\Thoma\eclipse-workspace\PyTorchClassify\classify>linear_relu_crossentropy.py  1000 1 .001 .5 2 20
Using cpu device
Model: NeuralNetwork(
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=10, out_features=20, bias=True)
    (1): ReLU()
    (2): Linear(in_features=20, out_features=20, bias=True)
    (3): ReLU()
    (4): Linear(in_features=20, out_features=4, bias=True)
  )
)
Display the class distributions
Begin training the model: 2026-11-01 15:31:03.422034
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************

Done training: 2026-11-01 15:44:31.053964
Elapsed time: 0:13:27.631930
Display the Cross Entropy Loss
Saved PyTorch Model State to data\linearReLUcrossEntropy.pth
Begin testing the model: 2026-11-01 15:46:11.080748

Done testing.  2026-11-01 15:46:11.157550
Elapsed time: 0:00:00.076802 Test Results:
|--------|--------|--------|--------|
|Class   |Count   |Correct |%Correct|
|--------|--------|--------|--------|
|1       |248     |144     |58      |
|--------|--------|--------|--------|
|2       |269     |193     |71      |
|--------|--------|--------|--------|
|3       |244     |230     |94      |
|--------|--------|--------|--------|
|4       |239     |72      |30      |
|--------|--------|--------|--------|
|Totals  |1000    |639     |64      |
|--------|--------|--------|--------|
</pre>
<h4>Class Property Distribution, Uniform, 10 properties, 4 classes</h4>
<img width="640" height="557" alt="classpropertydistr1000_3_15" src="https://github.com/user-attachments/assets/40e9ca72-a868-4849-9071-a7d017af357a" />
<h4>Learning Curve, 1000 batches, batch size = 1, learning rate = .001, momentum = 0.5, hidden layers = 2, features = 20</h4>
<img width="635" height="555" alt="learningcurve1000_2_20" src="https://github.com/user-attachments/assets/ae8ce024-fcc6-4fe5-bd3a-df497513b3a4" />
<pre>
C:\Users\Thoma\eclipse-workspace\PyTorchClassify\classify>linear_relu_crossentropy.py 1000 1 .001 .5 3 15
Using cpu device
Model: NeuralNetwork(
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=10, out_features=15, bias=True)
    (1): ReLU()
    (2): Linear(in_features=15, out_features=15, bias=True)
    (3): ReLU()
    (4): Linear(in_features=15, out_features=15, bias=True)
    (5): ReLU()
    (6): Linear(in_features=15, out_features=4, bias=True)
  )
)
Display the class distributions
Begin training the model: 2026-11-02 09:15:54.189164
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************
**************************************************

Done training: 2026-11-02 09:28:44.556764
Elapsed time: 0:12:50.367600
Display the Cross Entropy Loss
Saved PyTorch Model State to data\linearReLUcrossEntropy.pth
Begin testing the model: 2026-11-02 09:33:48.318506

Done testing.  2026-11-02 09:33:48.418498
Elapsed time: 0:00:00.099992 Test Results:
|--------|--------|--------|--------|
|Class   |Count   |Correct |%Correct|
|--------|--------|--------|--------|
|1       |252     |164     |65      |
|--------|--------|--------|--------|
|2       |234     |162     |69      |
|--------|--------|--------|--------|
|3       |240     |186     |77      |
|--------|--------|--------|--------|
|4       |274     |272     |99      |
|--------|--------|--------|--------|
|Totals  |1000    |784     |78      |
|--------|--------|--------|--------|  
</pre>
<h4>Class Property Distribution, Uniform, 10 properties, 4 classes</h4>
<img width="891" height="676" alt="image" src="https://github.com/user-attachments/assets/a5132da1-314e-418c-b31a-15348ddafa0b" />
<h4>Learning Curve, 1000 batches, batch size = 1, learning rate = .001, momentum = 0.5, hidden layers = 3, features = 15</h4>
<img width="638" height="555" alt="image" src="https://github.com/user-attachments/assets/e233cc56-92a8-42b3-a73e-d284b379e33c" />

