%Berke Oral 150130127
%This file is to test how effective k-NN classifier on test data sample
%
%Reads from modified csv file where first row removed and class values
%changed to numbers (class3 to 3)

clear;
train = load('train.csv');
shufTrain = train(randperm(size(train,1)),:);

trainSize = size(shufTrain, 1);
NumAtt = size(shufTrain, 2) - 2;
knnTestSize = 1000;
testNum = 1;

for redFeat = [20 35 50 65 80 93]
    dispText = sprintf('redFeat: %d, knnTestSize: %d', redFeat,knnTestSize);
    disp(dispText);
    TimerStart = tic;
    covTrain = cov(shufTrain(:,2:NumAtt+1));
    
    [eVec,eVal] = eigs(covTrain,redFeat,'LM');
    
    reducedTrain = shufTrain(:,2:NumAtt+1) *eVec;
    reducedTrain = [shufTrain(:,1) reducedTrain shufTrain(:,NumAtt+2)];
    
    %KNN1
    %Removing
    trainToTest = reducedTrain(1:knnTestSize,:);
    reduced2 = reducedTrain(knnTestSize + 1: trainSize, :);
    reduced2Size = trainSize - knnTestSize;
    
    totalClassification = 0;
    cor1 = 0;
    cor3 = 0;
    cor5 = 0;
    cor10 = 0;
    cor50 = 0;
    cor100 = 0;
    cor200 = 0;
    
    for k = 1 : knnTestSize
        for l = 1 : reduced2Size
            reduced2(l, redFeat + 3) = norm(reduced2(l, 2: redFeat+1) - trainToTest(k, 2:redFeat+1));
        end
        reduced2 = sortrows(reduced2,redFeat + 3);
        %Classifying
        totalClassification = totalClassification + 1;
        %--Knn3
        predictedClass = mode(reduced2(1:3, redFeat+2));
        if predictedClass == trainToTest(k, redFeat+2)
            cor3 = cor3 +1;
        end
        
        predictedClass = mode(reduced2(1:5, redFeat+2));
        if predictedClass == trainToTest(k, redFeat+2)
            cor5 = cor5 +1;
        end
        
        predictedClass = mode(reduced2(1:10, redFeat+2));
        if predictedClass == trainToTest(k, redFeat+2)
            cor10 = cor10 +1;
        end
        
        predictedClass = mode(reduced2(1:50, redFeat+2));
        if predictedClass == trainToTest(k, redFeat+2)
            cor50 = cor50 +1;
        end
        
        predictedClass = mode(reduced2(1:100, redFeat+2));
        if predictedClass == trainToTest(k, redFeat+2)
            cor100 = cor100 +1;
        end
        
        predictedClass = mode(reduced2(1:200, redFeat+2));
        if predictedClass == trainToTest(k, redFeat+2)
            cor200 = cor200 +1;
        end
        
        
        predictedClass = mode(reduced2(1, redFeat+2));
        if predictedClass == trainToTest(k, redFeat+2)
            cor1 = cor1 +1;
        end
    end
    
    acc1 = (cor1 / totalClassification)*100;
    acc3 = (cor3 / totalClassification)*100;
    acc5 = (cor5 / totalClassification)*100;
    acc10 = (cor10 / totalClassification)*100;
    acc50 = (cor50 / totalClassification)*100;
    acc100 = (cor100 / totalClassification)*100;
    acc200 = (cor200 / totalClassification)*100;
    TimePassed = toc(TimerStart);
    TestResult(testNum, :) = [redFeat knnTestSize acc1 acc3 acc5 acc10 acc50 acc100 acc200 TimePassed];
    testNum = testNum + 1;
end









