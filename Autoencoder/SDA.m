%% Supported in Matlab 2016

%% Reading Input
inputSize = 512;
X = zeros(512, 1250);
X = train_features;
filename = '/Users/ykg2910/Documents/4th_year_projects/Assignment3/training_labels.txt';
T = dlmread(filename);

%% Auto encoder 1
hiddenSize = 400;
autoenc1 = trainAutoencoder(X,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin');

features1 = encode(autoenc1,X);

%% Auto encoder 2
hiddenSize = 250;
autoenc2 = trainAutoencoder(features1,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin',...
    'ScaleData',false);

features2 = encode(autoenc2,features1);

%% Auto encoder 3
hiddenSize = 100;
autoen3 = trainAutoencoder(features2,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin',...
    'ScaleData',false);

features3 = encode(autoenc3,features2);

%% classification

softnet = trainSoftmaxLayer(features3,T,'LossFunction','crossentropy'); %softmax layer
deepnet = stack(autoenc1,autoenc2,autoenc3, softnet); %stacking of autoencoder
deepnet = train(deepnet,X,T); %training of SDA

wine_type = deepnet(X);
plotconfusion(T,wine_type);