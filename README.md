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
<h4>Learning Curve, 1000 batches, batch size = 1, learning rate = .001, momentum = 0.5, hidden layers = 2s, features = 20</h4>
<img width="635" height="555" alt="learningcurve1000_2_20" src="https://github.com/user-attachments/assets/ae8ce024-fcc6-4fe5-bd3a-df497513b3a4" />
