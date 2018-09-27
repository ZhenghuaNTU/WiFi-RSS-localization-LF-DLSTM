%% Generate Fingerprint dataset
% We have nine routers in the research lab environment

clear
clc
rand('seed',1);

load('refLoc.mat') % location of reference points
testIndex = [3 10 24 33 45 52 67 82 95 101]; % testing index for evaluation

L1 = [0 0]; % location of each router
L2 = [1 14];
L3 = [7 8.5];
L4 = [15 1];
L5 = [15 14];
L6 = [16 1];
L7 = [23.5 7.5];
L8 = [33 2];
L9 = [34 13];

TimeWin = 50; 
numOfrou = 9;

trainData = [];
testData = [];
trainTarget = [];
testTarget = [];


L = [L1 ; L2 ; L3; L4; L5; L6; L7; L8; L9]
% path loss model: P = P0 - 10 r log(d) + x
P0 = round(-60 + 30*rand(1,10)); % P0 from -60 to -30
r = 3.0 + 3*rand(1,10); % path loss parameter from 3.0 to 6.0
x = 3.0 + 3*rand(1,10); % random (caused by multi-path, ...) from 3.0 to 6.0

for numOfref = 1:length(refLoc)
    
    RSS = zeros(2000, length(L)); % generate data
    for ii = 1:2000% 2000 scaning
        for jj = 1:length(L)
            d = norm(refLoc(numOfref,:)-L(jj,:)); % distance
            RSS_one = round(P0(jj) - 10*r(jj)*log10(d) + x(jj)*randn);
            if RSS_one < -100
                RSS_one = -100;
            end
            RSS(ii,jj) = RSS_one;

        end
    end
    allData = RSS;
    
    
    if length(find(testIndex == numOfref)) == 0 % divided into training and testing
        
        for j = 1:size(allData,1)/TimeWin 
            temp = allData((j-1)*TimeWin+1:j*TimeWin,:); % sliding window
            trainData = [trainData ;temp];
            trainTarget = [trainTarget ; refLoc(numOfref,:)];
        end
    else
        for j = 1:size(allData,1)/TimeWin
            temp = allData((j-1)*TimeWin+1:j*TimeWin,:);
            testData = [testData ;  temp];
            testTarget = [testTarget ; refLoc(numOfref,:)];
        
        end
        
    end
end

%% normalization

alltemp = [trainData ;  testData];
min_v = repmat(min(alltemp),length(alltemp),1);
max_v = repmat(max(alltemp),length(alltemp),1);
alltemp = 2*(alltemp - min_v)./(max_v - min_v) - 1;
% alltemp = mapminmax([trainData ;  testData]');
% alltemp = alltemp';
trainData = alltemp(1:end-size(testData,1),:);
testData = alltemp(end-size(testData,1)+1:end,:);
trainData_temp = trainData;
testData_temp = testData;
trainData = zeros(size(trainTarget,1),TimeWin,numOfrou);
testData = zeros(size(testTarget,1),TimeWin,numOfrou);

for i = 1:size(trainTarget,1)
    trainData(i,:,:) = trainData_temp((i-1)*TimeWin+1:i*TimeWin,:);
end

for i = 1:size(testTarget,1)
    testData(i,:,:) = testData_temp((i-1)*TimeWin+1:i*TimeWin,:);
end
%  trainData = reshape(trainData,[3680,TimeWin*numOfrou]); %for normal algs
%  testData = reshape(testData,[400,TimeWin*numOfrou]);

%% Local features quantiles 0.25 0.5 0.75 min max 
numOffeature = 5;
trainData_temp = trainData;
testData_temp = testData;
smoothOrder = 10;

trainData = zeros(size(trainTarget,1),TimeWin/smoothOrder,numOfrou*numOffeature);
testData = zeros(size(testTarget,1),TimeWin/smoothOrder,numOfrou*numOffeature);

for i = 1:size(trainData,1)
    for j = 1:size(trainData_temp,3)
        temp = trainData_temp(i,:,j);
        for k = 1:size(trainData,2)
            quantile_temp = quantile(temp((k-1)*smoothOrder+1:k*smoothOrder),[0.25 0.50 0.75]);
            min_temp = min(temp((k-1)*smoothOrder+1:k*smoothOrder));
            max_temp = max(temp((k-1)*smoothOrder+1:k*smoothOrder));
            trainData(i,k,(j-1)*numOffeature+1:j*numOffeature) = [quantile_temp'; min_temp; max_temp];
        end
        
    end
end

for i = 1:size(testData,1)
    for j = 1:size(testData_temp,3)
        temp = testData_temp(i,:,j);
        for k = 1:size(testData,2)
            quantile_temp = quantile(temp((k-1)*smoothOrder+1:k*smoothOrder),[0.25 0.50 0.75]);
            min_temp = min(temp((k-1)*smoothOrder+1:k*smoothOrder));
            max_temp = max(temp((k-1)*smoothOrder+1:k*smoothOrder));
            testData(i,k,(j-1)*numOffeature+1:j*numOffeature) = [quantile_temp'; min_temp; max_temp];
        end
        
    end
end

save SimData1 trainData testData trainTarget testTarget % save data for evaluation 