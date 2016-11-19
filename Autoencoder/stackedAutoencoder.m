%% Training a Deep Neural Network 
% This example shows how to use the Neural Network Toolbox(TM) to train a
% deep neural network to classify images 

% This Algorithm train in greedy manner layer by layer training one layer at a time. 
% This example shows you how to train a neural network with two hidden
% layers along with a softmax layer


%% Data Set
inputSize = 512;
filename = '/Users/ykg2910/Documents/4th_year_projects/Assignment3/training_feature.txt';
xTrain = dlmread(filename,' ',1, 0);
xTrain  = transpose(xTrain);

%% Training the first Autoencoder
% You will begin by training a sparse autoencoder on the training data
% without using the labels(unsupervised training)
%
% An autoencoder is a neural network which attempts to replicate its input
% at its output. Thus, the size of its input will be the same as the size
% of its output. When the number of neurons in the hidden layer is less
% than the size of the input, the autoencoder will learn a compressed
% representation of the input.
%
% You can create an autoencoder by creating a feed-forward network, and
% then modifying some of the settings.

% Set the size of the hidden layer for the autoencoder. For the autoencoder
% that you are going to train, it is a good idea to make this smaller than
% the input size.
hiddenSize1 = 350;

% Create the network. You can experiment by changing the number of training
% epochs, and the training function
autoenc1 = feedforwardnet(hiddenSize1);
autoenc1.trainFcn = 'trainscg';
autoenc1.trainParam.epochs = 10000;

% Do not use process functions at the input or output
autoenc1.inputs{1}.processFcns = {};
autoenc1.outputs{2}.processFcns = {};

% Set the transfer function for both layers to the logistic sigmoid
autoenc1.layers{1}.transferFcn = 'logsig';
autoenc1.layers{2}.transferFcn = 'logsig';

% Use all of the data for training
autoenc1.divideFcn = 'dividetrain';

%%
% You can add regularizers that will encourage the autoencoder to learn a
% sparse representation in the first layer. By using the performance
% function |msesparse| you can control the influence of these regularizers
% by setting various parameters:
%
% * |L2WeightRegularization| controls the weighting of an L2 regularizer
% for the weights of the network (and not the biases). This should
% typically be quite small.
% * |sparsityRegularization| controls the weighting of a sparsity
% regularizer, which discourages large fractions of the neurons in the
% hidden layer from activating in response to an input.
% * |sparsity| controls the desired fraction of neurons that should
% activate in the first layer in response to an input. This must be
% between 0 and 1. The ideal value will vary depending on the nature of the
% problem.

autoenc1.performFcn = 'msesparse';

autoenc1.performParam.L2WeightRegularization = 0.004;
autoenc1.performParam.sparsityRegularization = 4;
autoenc1.performParam.sparsity = 0.15;

%%
% You now train the autoencoder. It should be noted that for an
% autoencoder, the input data and target data are identical.

% Train the autoencoder
autoenc1 = train(autoenc1,xTrain,xTrain);

%%
% You can view a diagram of the autoencoder, which shows the size of the
% input, output and hidden layer, as well as the transfer functions for the
% two layers.

view(autoenc1);

%% Visualizing the Results from the first Autoencoder
% After training the autoencoder, you can gain an insight into the features
% it has learned by visualizing them. Each neuron in the hidden layer will
% have a vector of weights associated with it in the input layer which will
% be tuned to respond to a particular visual feature. By reshaping these
% weight vectors, we can view a representation of these features.

W1 = autoenc1.IW{1};
% weightsImage = helperWeightsToImageGallery(W1,imageHeight,imageWidth,10,10);
% imshow(weightsImage);

%%
% It can be seen that the features learned by the autoencoder represent
% curls and stroke patterns from the digit images.
%
% The 100 dimensional output from the hidden layer of the autoencoder is a
% compressed version of the input, which summarizes its response to the
% features that were visualized above. Train the next autoencoder on a set
% of these vectors extracted from the training data. To do this, first
% create a version of the autoencoder with the final layer removed. This is
% done by creating an empty network object, and then manually configuring
% the settings. You copy the weights and biases from the trained
% autoencoder.

% Create an empty network
autoencHid1 = network;

% Set the number of inputs and layers
autoencHid1.numInputs = 1;
autoencHid1.numlayers = 1;

% Connect the 1st (and only) layer to the 1st input, and also connect the
% 1st layer to the output
autoencHid1.inputConnect(1,1) = 1;
autoencHid1.outputConnect = 1;

% Add a connection for a bias term to the first layer
autoencHid1.biasConnect = 1;

% Set the size of the input and the 1st layer
autoencHid1.inputs{1}.size = inputSize;
autoencHid1.layers{1}.size = hiddenSize1;

% Use the logistic sigmoid transfer function for the first layer
autoencHid1.layers{1}.transferFcn = 'logsig';

% Copy the weights and biases from the first layer of the trained
% autoencoder to this network
autoencHid1.IW{1,1} = autoenc1.IW{1,1};
autoencHid1.b{1,1} = autoenc1.b{1,1};

%%
% By calling the |view| function, you can see that this network is
% equivalent to the first autoencoder, with the last layer removed.

view(autoencHid1);

%%
% You can now generate the features that will be used to train the second
% autoencoder. This is done by evaluating the truncated autoencoder on the
% training data.

feat1 = autoencHid1(xTrain);

%% Training the second Autoencoder
% After training the first autoencoder, you train the second autoencoder in
% a similar way. The main difference is that the training data is the
% features generated from the hidden layer of the previous autoencoder.
% Once again, you create a feed-forward network and then modify the
% settings.

% Create the network. You can experiment by changing the size of the hidden
% layer, the number of training epochs, and the training function
hiddenSize2 = 200;
autoenc2 = feedforwardnet(hiddenSize2);
autoenc2.trainFcn = 'trainscg';
autoenc2.trainParam.epochs = 6000;

% Do not use process functions at the input or output
autoenc2.inputs{1}.processFcns = {};
autoenc2.outputs{2}.processFcns = {};

% Set the transfer function for both layers to the logistic sigmoid
autoenc2.layers{1}.transferFcn = 'logsig';
autoenc2.layers{2}.transferFcn = 'logsig';

% Use all of the data for training
autoenc2.divideFcn = 'dividetrain';

%%
% After creating the network, you set the performance function to
% |msesparse|, and set the values for the performance function parameters.

% Use the mean squared error with L2 weight and sparsity regularizers for
% the performance function
autoenc2.performFcn = 'msesparse';

% You can experiment by altering these parameters
autoenc2.performParam.L2WeightRegularization = 0.002;
autoenc2.performParam.sparsityRegularization = 4;
autoenc2.performParam.sparsity = 0.1;

%%
% Next, you train this autoencoder on the features generated from the
% previous autoencoder.

% Train the second autoencoder
autoenc2 = train(autoenc2,feat1,feat1);

%%
% Once again, you can view a diagram of the autoencoder by calling the
% |view| command. The second autoencoder is similar to the first, but the
% size of the layers are different.

view(autoenc2);

%%
% As before, you create a version of the second autoencoder with the final
% layer removed.

% Create an empty network
autoencHid2 = network;

% Set the number of inputs and layers
autoencHid2.numInputs = 1;
autoencHid2.numlayers = 1;

% Connect the 1st (and only) layer to the 1st input, and also connect the
% 1st layer to the output
autoencHid2.inputConnect(1,1) = 1;
autoencHid2.outputConnect = 1;

% Add a connection for a bias term to the first layer
autoencHid2.biasConnect = 1;

% Set the size of the input and the 1st layer
autoencHid2.inputs{1}.size = hiddenSize1;
autoencHid2.layers{1}.size = hiddenSize2;

% Use the logistic sigmoid transfer function for the first layer
autoencHid2.layers{1}.transferFcn = 'logsig';

% Copy the weights and biases from the first layer of the second trained
% autoencoder to this network
autoencHid2.IW{1,1} = autoenc2.IW{1,1};
autoencHid2.b{1,1} = autoenc2.b{1,1};

%%
% You can call the |view| function to see a diagram of this network. It is
% equivalent to the second autoencoder with the last layer removed.
W2 = autoenc2.IW{1};
view(autoencHid2);

%%
% You can extract a second set of features by passing the previous set
% through the second truncated autoencoder.

feat2 = autoencHid2(feat1);

%% Training the third Autoencoder
% After training the first autoencoder, you train the second autoencoder in
% a similar way. The main difference is that the training data is the
% features generated from the hidden layer of the previous autoencoder.
% Once again, you create a feed-forward network and then modify the
% settings.

% Create the network. You can experiment by changing the size of the hidden
% layer, the number of training epochs, and the training function
hiddenSize3 = 100;
autoenc3 = feedforwardnet(hiddenSize3);
autoenc3.trainFcn = 'trainscg';
autoenc3.trainParam.epochs = 5000;

% Do not use process functions at the input or output
autoenc3.inputs{1}.processFcns = {};
autoenc3.outputs{2}.processFcns = {};

% Set the transfer function for both layers to the logistic sigmoid
autoenc3.layers{1}.transferFcn = 'logsig';
autoenc3.layers{2}.transferFcn = 'logsig';

% Use all of the data for training
autoenc3.divideFcn = 'dividetrain';

%%
% After creating the network, you set the performance function to
% |msesparse|, and set the values for the performance function parameters.

% Use the mean squared error with L2 weight and sparsity regularizers for
% the performance function
autoenc3.performFcn = 'msesparse';

% You can experiment by altering these parameters
autoenc3.performParam.L2WeightRegularization = 0.002;
autoenc3.performParam.sparsityRegularization = 4;
autoenc3.performParam.sparsity = 0.1;

%%
% Next, you train this autoencoder on the features generated from the
% previous autoencoder.

% Train the second autoencoder
autoenc3 = train(autoenc3,feat2,feat2);

%%
% Once again, you can view a diagram of the autoencoder by calling the
% |view| command. The second autoencoder is similar to the first, but the
% size of the layers are different.

view(autoenc3);

%%
% As before, you create a version of the second autoencoder with the final
% layer removed.

% Create an empty network
autoencHid3 = network;

% Set the number of inputs and layers
autoencHid3.numInputs = 1;
autoencHid3.numlayers = 1;

% Connect the 1st (and only) layer to the 1st input, and also connect the
% 1st layer to the output
autoencHid3.inputConnect(1,1) = 1;
autoencHid3.outputConnect = 1;

% Add a connection for a bias term to the first layer
autoencHid3.biasConnect = 1;

% Set the size of the input and the 1st layer
autoencHid3.inputs{1}.size = hiddenSize2;
autoencHid3.layers{1}.size = hiddenSize3;

% Use the logistic sigmoid transfer function for the first layer
autoencHid3.layers{1}.transferFcn = 'logsig';

% Copy the weights and biases from the first layer of the second trained
% autoencoder to this network
autoencHid3.IW{1,1} = autoenc3.IW{1,1};
autoencHid3.b{1,1} = autoenc3.b{1,1};

%%
% You can call the |view| function to see a diagram of this network. It is
% equivalent to the second autoencoder with the last layer removed.
W3 = autoenc3.IW{1};
view(autoencHid3);

%%
% You can extract a second set of features by passing the previous set
% through the second truncated autoencoder.

feat3 = autoencHid3(feat2);

%% Training the final Softmax Layer
% You will create a softmax layer, and train it on the output from the
% hidden layer of the second autoencoder. As the softmax layer only
% consists of one layer, you create it manually.

% Create an empty network
finalSoftmax = network;

% Set the number of inputs and layers
finalSoftmax.numInputs = 1;
finalSoftmax.numLayers = 1;

% Connect the 1st (and only) layer to the 1st input, and connect the 1st
% layer to the output
finalSoftmax.inputConnect(1,1) = 1;
finalSoftmax.outputConnect = 1;

% Add a connection for a bias term to the first layer
finalSoftmax.biasConnect = 1;

% Set the size of the input and the 1st layer
finalSoftmax.inputs{1}.size = hiddenSize3;
finalSoftmax.layers{1}.size = 5;

% Use the softmax transfer function for the first layer
finalSoftmax.layers{1}.transferFcn = 'softmax';

% Use all of the data for training
finalSoftmax.divideFcn = 'dividetrain';

% Use the cross-entropy performance function
finalSoftmax.performFcn = 'crossentropy';

% You can experiment by the number of training epochs and the training
% function
finalSoftmax.trainFcn = 'trainscg';
finalSoftmax.trainParam.epochs = 5000;

%%
% Next, you train the softmax layer. Unlike the autoencoders, you will
% train the softmax layer in a supervised fashion using labels for the
% training data.

%% Creating groundtruth 
filename = '/Users/ykg2910/Documents/4th_year_projects/Assignment3/training_labels.txt';
labels = dlmread(filename,' ',1, 0);
labels = transpose(labels);

tTrain = zeros(5, 1249);
    for i = 1 : 1249
        tTrain(labels(1, i),i) = 1.0;
    end;
finalSoftmax = train(finalSoftmax,feat3,tTrain);

%%
% You can view a diagram of the softmax layer by calling the |view|
% command.

view(finalSoftmax);

%% Forming a Multilayer Neural Network
% You have trained three separate components of a deep neural network in
% isolation. At this point, it maybe be useful to view these three
% components. They are the networks |autoencHid1|, |autoencHid2| and
% |finalSoftmax|.

view(autoencHid1);
view(autoencHid2);
view(autoencHid3);
view(finalSoftmax);

%%
% You join these layers together to form a multilayer neural network. You
% create the neural network manually, and then configure the settings, and
% copy the weights and biases from the autoencoders and softmax layer.

% Create an empty network
finalNetwork = network;

% Specify one input and three layers
finalNetwork.numInputs = 1;
finalNetwork.numLayers = 4;

% Connect the 1st layer to the input
finalNetwork.inputConnect(1,1) = 1;

% Connect the 2nd layer to the 1st layer
finalNetwork.layerConnect(2,1) = 1;

% Connect the 3rd layer to the 2nd layer
finalNetwork.layerConnect(3,2) = 1;

% Connect the 4th layer to the 3rd layer
finalNetwork.layerConnect(4,3) = 1;

% Connect the output to the 4th layer
finalNetwork.outputConnect(1,4) = 1;

% Add a connection for a bias term for each layer
finalNetwork.biasConnect = [1; 1; 1; 1];

% Set the size of the input
finalNetwork.inputs{1}.size = inputSize;

% Set the size of the first layer to the same as the layer in autoencHid1
finalNetwork.layers{1}.size = hiddenSize1;

% Set the size of the second layer to the same as the layer in autoencHid2
finalNetwork.layers{2}.size = hiddenSize2;

% Set the size of the third layer to the same as the layer in autoencHid2
finalNetwork.layers{3}.size = hiddenSize3;

% Set the size of the forth layer to the same as the layer in finalSoftmax
finalNetwork.layers{4}.size = 5;

% Set the transfer function for the first layer to the same as in
% autoencHid1
finalNetwork.layers{1}.transferFcn = 'logsig';

% Set the transfer function for the second layer to the same as in
% autoencHid2
finalNetwork.layers{2}.transferFcn = 'logsig';

% Set the transfer function for the second layer to the same as in
% autoencHid3
finalNetwork.layers{3}.transferFcn = 'logsig';

% Set the transfer function for the third layer to the same as in
% finalSoftmax
finalNetwork.layers{4}.transferFcn = 'softmax';

% Use all of the data for training
finalNetwork.divideFcn = 'dividetrain';

% Copy the weights and biases from the three networks that have already
% been trained
finalNetwork.IW{1,1} = autoencHid1.IW{1,1};
finalNetwork.b{1} = autoencHid1.b{1,1};
finalNetwork.LW{2,1} = autoencHid2.IW{1,1};
finalNetwork.b{2} = autoencHid2.b{1,1};
finalNetwork.LW{3,2}= autoencHid3.IW{1,1};
finalNetwork.b{3} = autoencHid3.b{1,1};
finalNetwork.LW{4,3} = finalSoftmax.IW{1,1};
finalNetwork.b{4} = finalSoftmax.b{1,1};

% Use the cross-entropy performance function
finalNetwork.performFcn = 'crossentropy';

% You can experiment by changing the number of training epochs and the
% training function
finalNetwork.trainFcn = 'trainscg';
finalNetwork.trainParam.epochs = 10000;

%%
% You can view a diagram of the multilayer network with the |view| command.
view(finalNetwork);

%%
% With the full deep network formed, you can compute the results on the
% test set. Before you can do this, you have to reshape the test images
% into a matrix, as was done for the training set.

% Load the test images
filename = '/Users/ykg2910/Documents/4th_year_projects/Assignment3/test_feature.txt';
xTest = dlmread(filename,' ',1, 0);
xTest  = transpose(xTest);

%%
% You can visualize the results with a confusion matrix. The numbers in the
% bottom right hand square of the matrix will give the overall accuracy.

y = finalNetwork(xTest);


%% Fine tuning the Deep Neural Network
% The results for the deep neural network can be improved by performing
% backpropagation on the whole multilayer network. This process is often
% referred to as fine tuning.
%
% You fine tune the network by retraining it on the training data in a
% supervised fashion. You then view the results again using a confusion
% matrix.

finalNetwork = train(finalNetwork,xTrain,tTrain);
y = finalNetwork(xTest);

filename = '/Users/ykg2910/Documents/4th_year_projects/Assignment3/test_labels.txt';
labels = dlmread(filename,' ',1, 0);
labels = transpose(labels);

tTest = zeros(5, 249);
    for i = 1 : 249
        tTest(labels(1, i),i) = 1.0;
    end;
    
plotconfusion(tTest,y);

%% Summary
% This example showed how to train a deep neural network to classify digits
% in images using the Neural Network Toolbox(TM). The steps that have been
% outlined could be applied to other similar problems such as classifying
% images of letters, or even small images of objects of a specific
% category.

displayEndOfDemoMessage(mfilename)