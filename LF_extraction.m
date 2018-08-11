%% quantiles 0.25 0.5 0.75 min max 

load 'rawData.mat' 
% load original training and testing data three dimentional: (NumOfsamples,Time steps, dimention)

TimeWin = 50; % original size of the sequence
numOffeature = 5; % number of features
trainData_temp = trainData; % training data three dimentional: (NumOfsamples,Time steps, dimention)
testData_temp = testData;
smoothOrder = 10; % window size
numOfrou = 9; % number of routers (original feature dimention)

trainData = zeros(size(trainTarget,1),TimeWin/smoothOrder,numOfrou*numOffeature);
testData = zeros(size(testTarget,1),TimeWin/smoothOrder,numOfrou*numOffeature);

for i = 1:size(trainData,1)
    for j = 1:size(trainData_temp,3)
        temp = trainData_temp(i,:,j);
        for k = 1:size(trainData,2)
            %trainData(i,k,j) = mean(temp((k-1)*smoothOrder+1:k*smoothOrder));
            
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
            %testData(i,k,j) = mean(temp((k-1)*smoothOrder+1:k*smoothOrder));
            quantile_temp = quantile(temp((k-1)*smoothOrder+1:k*smoothOrder),[0.25 0.50 0.75]);
            min_temp = min(temp((k-1)*smoothOrder+1:k*smoothOrder));
            max_temp = max(temp((k-1)*smoothOrder+1:k*smoothOrder));
            testData(i,k,(j-1)*numOffeature+1:j*numOffeature) = [quantile_temp'; min_temp; max_temp];
        end
        
    end
end

% the final output for training and testing data has a shape of (NumofSamples, 50/10 steps, numOfrou*numOffeature)
