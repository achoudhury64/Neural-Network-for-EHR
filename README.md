# Neural-Network-for-EHR
Neural Networks and Deep Learning are becoming more popular and widely applied to many data science domain including healthcare applications.
We will use Cross Entropy loss function. 

Epileptic Seizure Classification
we will use Epileptic Seizure Recognition Data Set which is originally from UCI Machine Learning Repository. 

Data Loading
First of all, we need to load the dataset to train our models. Basically, you will load the raw dataset files, and convert them into a PyTorch TensorDataset which contains data tensor and target (label) tensor. Since each model requires a different shape of input data, we should convert the raw data accordingly.

Multi-layer Perceptron
We start with a simple form of neural network model first, Multi-layer Perceptron (MLP).
First we Complete a class MyMLP in mymodels.py, and implement a 3-layer MLP , then Use a hidden layer composed by 16 hidden units, followed by a sigmoid activation function. The function plot learning curves in plots.py to plot loss curves and accuracy curves for training and validation sets. We  should control the number of epochs (in the main script) according to your model training. Once we  have implemented a very simple MLP model, we Try to improve it by changing model parameters, e.g., a number of layers and units, or applying any trick and technique applicable to neural networks.

Convolutional Neural Network
Our next model is CNN. Implement a CNN model constituted by a couple of convolutional layers, pooling layer, and fully-connected layers at the end. MyCNN class in mymodels.py uses  two convolutional layers, one with 6 filters of the kernel size 5 (stride 1) and the other one with 16 filters with the kernel size 5 (stride 1), and they must be followed by Rectified Linear Unit (ReLU) activation. Each convolutional layer (after ReLU activation) is followed by a max pooling layer with the size (as well as stride) of 2. There are two fully-connected (as known as dense or linear) layer, one with 128 hidden units followed by ReLU activation and the other one is the output layer that has five units. We Calculate the number of ”trainable” parameters in the model with providing the cal- culation details. 

Recurrent Neural Network
The final model we will implement on this dataset is RNN. For an initial architecture, you will use a Gated Recurrent Unit (GRU) followed by a fully connected layer. Since we have a sequence of inputs with a single label, we will use a many-to-one type of architecture. we have the mymodels.py. Use one GRU layer with 16 hidden units. There should be one fully-connected layer connecting the hidden units of GRU to the output units.
We calculate the number of ”trainable” parameters in the model with providing the cal- culation details. 

Mortality Prediction with RNN
In the previous problem, the dataset consists of the same length sequences. In many real- world problems, however, data often contains variable-length of sequences, natural language processing for example. Also in healthcare problems, especially for longitudinal health records, each patient has a different length of clinical records history. In this problem, we will apply a recurrent neural network on variable-length of sequences from longitudinal electronic health record (EHR) data. Dataset for this problem was extracted from MIMIC- III dataset. 

Preprocessing
preprocessing the dataset is one of the most impor- tant part of data science before you apply any kind of machine learning algorithm. Here, we will implement a pipeline that process the raw dataset to transform it to a structure that we can use with RNN model. You can use typical Python packages such as Pandas and Numpy.
Simplifying the problem, we will use the main codes, which are the first 3 or 4 alphanumeric values prior to the decimal point, of ICD-9 codes as features in this homework. For example, for the ICD-9 code E826.9 (Pedal cycle accident injuring unspecified person), we will use E826 only, and thus it will be merged with other sub-codes under E826 such as E826.4 (Pedal cycle accident injuring occupant of streetcar).
The pipeline procedure  can be summarized as follows.
1. Loading the raw data files.
2. Group the diagnosis codes given to the same patient on the same date.
3. Sorting the grouped diagnoses for the same patient, in chronological order.
4. Extracting the main code from each ICD-9 diagnosis code, and converting them into unique feature ids, 0 to d − 1, where d is the number of unique main codes.
5. Converting into the final format we want. Here we will use a List of (patient) List of (code) List as our final format of the dataset.

Creating Custom Dataset

PyTorch DataLoader will generate mini-batches for you once you give it a valid Dataset, and most times you do not need to worry about how it generates each mini-batch if your data have fixed size. In this problem, however, we use variable size (length) of data, e.g., visits of each patient are represented by a matrix with a different size as in the example above. Therefore, we have to specify how each mini-batch should be generated by defining collate fn, which is an argument of DataLoader constructor. Here, we will create a custom collate function named visit collate fn, which creates a mini-batch represented by a 3D Tensor that consists of matrices with the same number of rows (visits) by padding zero-rows at the end of matrices shorter than the largest matrix in the mini-batch. Also, the order of matrices in the Tensor must be sorted by the length of visits in descending order. In addition, Tensors contains the lengths and the labels are also have to be sorted accordingly.

Building Model
Now, we can define and train our model. In MyVariableRNN class in mymodels.py. First layer is a fully-connected layer with 32 hidden units, which acts like an embedding layer to project sparse high-dimensional inputs to dense low-dimensional space. The projected inputs are passed through a GRU layer that consists of 16 units, which learns temporal relationship. It is followed by output (fully-connected) layer constituted by two output units. In fact, we can use a single output unit with a sigmoid activation function since it is a binary classification problem. However, we use a multi-class classification setting with the two output units to make it consistent with the previous problem for reusing the utility functions we have used. Use tanh activation funtion after the first FC layer, and remind that you should not use softmax or sigmoid activation after the second FC (output) layer since the loss function will take care of it. Train the constructed model and evaluate it. Included all learning curves and the confusion matrix in the report.


