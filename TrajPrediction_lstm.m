clc;clear all;close all;

%synthetic reach data from LQG 2 joint controller
load('dat3.mat')
load netIMtoAng.mat
load netAngToPos.mat

YPred = predict(netIMtoAng,dat3);
Out1  = netAngToPos(YPred');

%% visualise  decoded data
for i= 1:160
    % plot path in joint space
    p1 =  Out1(1:51,i);
    p2 =  Out1(52:102,i);
    v1 =  Out1(103:153,i);
    v2 =  Out1(154:204,i);
    subplot(131);plot(p1,p2,'.k'); title('Pos');hold on
    subplot(132);plot(v1,v2,'.k'); title('Vel');
    hold on;
%     pause
    subplot(133); plot(tTra(3,i),tTra(4,i),'o'); hold on;
end
% axis image;
subplot(132);xlabel('shoulder (rad)');
ylabel('elbow (rad)');


%% Test
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M7.tif','r');
S1   = read(t);
s{7} = rgb2gray(S1);
B{7} = imresize(s{7},[28 28]);
TestImage(:,:,1,1) = B{7}(:,:);

YPred = predict(netIMtoAng,TestImage(:,:,1,1));
Out2  = netAngToPos(YPred');

subplot(141); imshow(S1);title('Cartesian Work space')
subplot(142); plot(tTra(3,7),tTra(4,7),'or'); hold on;
plot(pi/2,pi/2,'or'); hold on;
xlim([0.8,2.4]);ylim([0.8,2.4]);axis square;title('Joint space');
xlabel('shoulder (rad)');
subplot(143);
p1 =  Out2(1:51,1);
p2 =  Out2(52:102,1);
v1 =  Out2(103:153,1);
v2 =  Out2(154:204,1);
plot(p1,p2,'.k'); title('Pos');hold on;
xlim([0.8,2.4]);ylim([0.8,2.4]);
xlabel('shoulder (rad)');
ylabel('elbow (rad)');
axis square
subplot(144);plot(v1,v2,'.k'); title('Vel');
xlabel('shoulder (rad/s)');
ylabel('elbow (rad/s)');
xlim([-2.5,2.5]);ylim([-2.5,2.5]);
axis square

%% Lstm
data  = Out1(3:51,1)';
% dataY =  Out1(54:102,1)';
% velX  =  Out1(105:153,1)';
% velY  =  Out1(156:204,1)';
% DD    = [dataX; dataY; velX; velY];

figure
plot(data)

%% prepare input -output for lstm
numTimeStepsTrain = floor(0.9*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataTest  = data(numTimeStepsTrain+1:end);
mu        = mean(dataTrain);
sig       = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;
XTra = dataTrainStandardized(1:end-1);
YTra = dataTrainStandardized(2:end);

%% LSTM training
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTra,YTra,layers,options);
analyzeNetwork(net)
%% forcast- predicted trajectory
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
net = predictAndUpdateState(net,XTra);
[net,YPred] = predictAndUpdateState(net,YTra(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
YPred = sig*YPred + mu;
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))
figure
plot(dataTrain(1:end-1),'Linewidth',6)
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-','Markersize',32)
hold off
legend(["Observed" "Forecast"]);
set(gca,'Fontsize',20);
xlabel('Time steps');
ylabel('Shoulder angle position(rad)')