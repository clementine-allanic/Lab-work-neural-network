# Lab-work-neural-network

Training of a neural network 


To begin with, I wanted to find the optimal batch size so I ran a few calculation, changing only the batch size :

batch_size = 15, n_epoch = 10, learning_rate = 0.001 => 75.38  64.22  63.73
batch_size = 60, n_epoch = 10, learning_rate = 0.001 => 75.15  64.94  65.05
batch_size = 80, n_epoch = 10, learning_rate = 0.001 => 75.58  64.12  63.44
batch_size = 70, n_epoch = 10, learning_rate = 0.001 => 75.56  64.50  64.08
batch_size = 50, n_epoch = 10, learning_rate = 0.001 => 75.46  64.37  66.44
batch_size = 40, n_epoch = 10, learning_rate = 0.001 => 75.83  64.36  64.40

I decided to use a batch size of 50, since it was the hyperparameter that gave by far the best results on the test sample. However, I discovered after restarting the calculations with the same hyperparameters, that the results were slightly different.
However, it seemed a good compromise and the results were still slightly better.

Then, I experimented with the learning rate. Here are the results for different learning rate : 

batch_size = 50, n_epoch = 10, learning_rate = 0.0005 => 70.89 / 62.83 / 62.35
batch_size = 50, n_epoch = 10, learning_rate = 0.0007 => 75.72 / 63.79 / 63.09
batch_size = 50, n_epoch = 10, learning_rate = 0.001   => 75.46 / 64.37 / 66.44
batch_size = 50, n_epoch = 10, learning_rate = 0.0012 => 72.92 / 62.82 / 63.18
batch_size = 50, n_epoch = 10, learning_rate = 0.002   => 74.74 / 63.77 / 62.91

We have seen in the lecture that the batch size and the learning rate are connected so we can see that we have a better result with a learning rate of 0.001. Besides, it is also important to have a learning rate that will not engender any overfitting.
I decided to keep the learning rate at 0.001.

We then tried to see the influence of the number of epochs :

batch_size = 50, n_epoch = 4, learning_rate = 0.001 => 68.03 / 61.63 / 61.28
batch_size = 50, n_epoch = 8, learning_rate = 0.001 => 74.81 / 64.28 / 63.99
batch_size = 50, n_epoch = 10, learning_rate = 0.001 => 73.26 / 62.67 / 63.19
batch_size = 50, n_epoch = 12, learning_rate = 0.001 => 76.26 / 64.89 / 64.07
batch_size = 50, n_epoch = 16, learning_rate = 0.001 => 74.65 / 64.37 / 63.23

We notice that by using a greater number of epochs, the valuation loss start to
 slightly go up again so we have to be careful about overfitting. Besides, between the 10th and the 16th epochs, the training loss does not decrease that much, compared to the computation time added so it might not be worth it. However, as the number of epochs represent the number of times that the neural network will go over all the data, a number of epochs too low will not give really good results either.


We then start experimenting on the architecture of the neural network.
We start by adding layers of 18 filters each with the same setting :
[batch_size = 50, n_epoch = 10, learning_rate = 0.001]
We add convX = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1)
Between each layer we do a ReLU and a max pooling.

For 1 layer   : 75.42 / 66.78 / 66.74
For 2 layers : 72.55 / 67.52 / 67.66
For 3 layers : 67.02 / 64.16 / 64.58

We notice that for 3 layers, the results seem to be dropping. However, this is linked to the fact that the learning rate does not seem to be appropriate anymore, the valuation loss drops but never stagnates. The more layers we add, the more we have to increase the learning rate.
We keep the last architecture with 3 new layers - so a total of 4 layers. And increase the learning rate : 
For learning rate of 0.0012 : 69.01 / 65.78 / 64.60
For learning rate of 0.0015 : 69.01 / 65.45 / 64.77

The results are slightly better but they don’t seem to have a real influence on the results.

We return to an architecture of 3 layers (including the first one)- which gave the best results and try to experiment on the number of neurons on each layer. We keep the same number on each hidden layer.
The hyperparameters batch_size = 50, n_epoch = 10, learning_rate = 0.0015
For n_neurons = 40   :81.84 / 71.74 / 72.11
For n_neurons = 80   :84.19 / 73.79 / 73.33
For n_neurons = 90  : 86.77 / 74.10 / 74.00
For n_neurons = 100 :83.04 / 73.14 / 72.36
For n_neurons = 120 : 82.60 / 74.30 / 73.28
For n_neurons = 160 : 85.73 / 73.98 / 73.42

Starting from a number of neurons of around 80 we don’t notice any sizable difference in the results. However, the valuation loss for n=120 and n=160 starts to rise at the end, so we might be overfitting a bit.
I will keep n = 90 from now on.

With that in mind, I tried adding another layer with the same number of neuron to see if it would improve the result : 
Adding one layer : 79.45 / 71.62 / 71.55
So, as before, we will stay with a 3 layers structure.

We can experiment by modifying the kernel size of the layers :
We remind that the hyperparameters are batch_size = 50, n_epoch = 10, learning_rate = 0.0015 :
kernel_size = 1 : 61.88 / 57.64 / 58.27
kernel_size = 2 : 80.35 / 70.75 / 70.29
kernel_size = 3:  80.50 / 71.34 / 72.17
kernel_size = 4:  84.23 / 72.80 / 72.58
kernel_size = 5 : 77.96 / 68.89 / 68.78

On garde donc un kernel_size = 4 pour la suite des expériences;

On essaie de modifier la taille de la première fully connected layer pour voir son impact : 
fc_size = 30 : 78.47 / 71.38 / 71.58
fx_size = 60 : 78.81 / 70.45 / 70.73
fc_size = 80 : 82.20 / 73.36 / 72.57
fc_size = 90 : 85.87 / 73.15 / 72.91
fc_size = 100 : 84 / 74.41 / 72.56
fc_size = 120: 82.97 / 72.61 / 72.40

We fix the size of the first fully connected layer to 90.

We will then try to see how the repartition of the data between the training and validation set affect the results.
Pour learning = 45000 / validation = 5000 : 82.98 / 74.62 / 72.63
Pour learning = 48000 / validation = 2000 : 89.38 / 83.24 / 73.29

So it seems like increasing the number of learning images increases the success rate.
However, the learning rate seems to be lacking, let’s try increasing it a bit :

For learning_rate = 0.0017 : 86.50 / 82.60 / 73.57
For learning_rate = 0.0018 : 85.41 / 81.12 / 71.76
So increasing the learning_rate might not be as useful as I thought.

This lab work was really useful in understanding how the different parameters influence each other and how this can affect the success rate of the neural network. I found trying to get the best success rate as possible really fun. However, the results could widely vary from an iteration to the other so sometimes finding the optimal solution could be difficult. 

The neural network currently in the pytorch file is the result of fooling around so i did not take notes of the modification that I did, however the best success rate I ever got on the test set with it was 74.39%.
