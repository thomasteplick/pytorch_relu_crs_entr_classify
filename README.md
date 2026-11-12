<h3>PyTorch Neural Network Classification using Linear Layers, ReLU Activation Functions, Stochastic Gradient Descent (SGD), and Cross Entropy Loss</h3>
<p>
This program classifies data from four uniformly-distributed sources.  Each source has ten properties that are uniformly distributed in a sub-interval of (0, 256].
Input samples are randomly taken from the ten regions for each class and are used as the input to the neural network after normalization.  The normalization simply
divides the input by 256 which produces a floating point number in [0,1].  The target is the specific class as an 
integer in (0, 3).  This is termed one-hot vector since only one of the four outputs is one and the rest are zeros; for example, a class 3 target would be [0,0,1,0].
The user can specify 1, 2, or 3 hidden layers with up to 20 features in each hidden layer.  So the input samples consist of the ten property values and the 
target integer.  The user can specifiy the number of epochs and the epoch sample size.  The user can also specify the learning rate and momentum hyperparameters.
After the network is trained, the Cross Entropy Loss per epoch is displayed.  The testing phase follows the training.  A different set of samples is presented to
the neural network.  The number of sample inputs for each class and the number of correct decisions is recorded.  The test results for each class and the totals 
are shown in a table.  The activation function used for each fully-connected layer is Rectified Linear Unit (ReLU).  The Cross Entropy Loss function is used to determine
the output error and backpropagation is used to distribute the error to previous-layer weights and biases using partial derivatives and the chain rule of 
Differential Calculus.  PyTorch handles all this transparently to the user with Tensors and Autograd.  The Cross Entropy Loss function uses Softmax to estimate
the probabilites of each output class and then takes the negative log of these values.  The maximum of these outputs is the predicted class.  Again this is 
done under the covers; the library handles all the details.
</p>
<p>
The classification results vary depending on the class property distribution.  It is hard to separate classes whose distributions overlap one another.  And the degree to
which each property can vary (its standard deviation) makes it harder to predict.  When the batch size was increased, some of the classes %correct went to zero.
When the batch size was decreased to one, those classes %correct increased.  The large batch sizes smooth the learning curve variation but decreases resolution
and some classes are harder to detect due to their degree of overlap with other classes.  The weights and biases are not corrected during backpropagation for every samples when the batch
size is greater than one.  This means that the weight and bias corrections are done less frequently.  Increasing the number of batches with the batch size set to one does 
increase the %correct up to a certain value and then it levels off with no further increase.
</p>
<p>
This program uses PyTorch (without a GPU), Python 3.13, Matplotlib.Py, and Numpy.
</p>

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
<h4>Class Property Distribution, Uniform, 10 properties, 4 classes</h4>
<img width="925" height="610" alt="image" src="https://github.com/user-attachments/assets/907f5553-57ba-405e-83f9-53b61208bb32" />
<pre>
C:\Users\Thoma\eclipse-workspace\PyTorchClassify\classify>linear_relu_crossentropy.py 1000 1 .001 .5 2 20
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
Begin training the model: 2026-11-12 16:22:16.935931
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

Done training: 2026-11-12 16:35:29.265209
Elapsed time: 0:13:12.329278
Display the Cross Entropy Loss
Saved PyTorch Model State to data\linearReLUcrossEntropy.pth
Begin testing the model: 2026-11-12 16:37:14.262858

Done testing.  2026-11-12 16:37:14.338928
Elapsed time: 0:00:00.076070 Test Results:
|--------|--------|--------|--------|
|Class   |Count   |Correct |%Correct|
|--------|--------|--------|--------|
|1       |261     |261     |100     |
|--------|--------|--------|--------|
|2       |229     |229     |100     |
|--------|--------|--------|--------|
|3       |238     |238     |100     |
|--------|--------|--------|--------|
|4       |272     |272     |100     |
|--------|--------|--------|--------|
|Totals  |1000    |1000    |100     |
|--------|--------|--------|--------|
</pre>
<h4>Learning Curve, 1000 batches, batch size = 1, learning rate = .001, momentum = 0.5, hidden layers = 2, features = 20</h4>
<img width="639" height="551" alt="image" src="https://github.com/user-attachments/assets/291d8ecf-29e8-4779-8a1a-3b3b772448c3" />
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
Begin training the model: 2026-11-12 16:39:13.572212
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

Done training: 2026-11-12 16:55:11.329350
Elapsed time: 0:15:57.757138
Display the Cross Entropy Loss
Saved PyTorch Model State to data\linearReLUcrossEntropy.pth
Begin testing the model: 2026-11-12 16:56:34.000548

Done testing.  2026-11-12 16:56:34.090103
Elapsed time: 0:00:00.089555 Test Results:
|--------|--------|--------|--------|
|Class   |Count   |Correct |%Correct|
|--------|--------|--------|--------|
|1       |247     |247     |100     |
|--------|--------|--------|--------|
|2       |256     |256     |100     |
|--------|--------|--------|--------|
|3       |248     |248     |100     |
|--------|--------|--------|--------|
|4       |249     |249     |100     |
|--------|--------|--------|--------|
|Totals  |1000    |1000    |100     |
|--------|--------|--------|--------|
</pre>
<h4>Learning Curve, 1000 batches, batch size = 1, learning rate = .001, momentum = 0.5, hidden layers = 3, features = 15</h4>
<img width="633" height="550" alt="image" src="https://github.com/user-attachments/assets/359ab244-f8b0-4423-adb7-10218ebbc2c1" />

