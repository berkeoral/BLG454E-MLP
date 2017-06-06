%Berke Oral 150130127
%Makes MLP classification for final result
%
%Reads from modified csv files where first row removed and class values
%changed to numbers (class3 to 3)

clear;
train = load('train.csv');
test = load('test.csv');
testSize = size(test, 1);

shufTrain = train(randperm(size(train,1)),:);
trainSize = size(shufTrain, 1);

shufTrain = [shufTrain(:,1) ones(trainSize,1) shufTrain(:, 2:95)];
NumAtt = size(shufTrain, 2) - 3;
test = [test(:,1) ones(testSize,1) test(:, 2:94)];

hiddenLayerSize = 64;
n = 0.005;
itNum = 10;
K = 9;

initWeightInterval = 0.05;
dispText = sprintf('Traning');
disp(dispText);
dispText = sprintf('hiddenLayerSize:%d, n:%d, itNum:%d', hiddenLayerSize, n, itNum);
disp(dispText);
%initial random weights
for count = 1 : hiddenLayerSize
    for count2 = 1 : NumAtt +1
        W(count,count2) = (rand(1) * initWeightInterval * 2) - initWeightInterval; % W weight vectors of Hidden layer
    end
end
for count = 1 : K
    for count2 = 1 : hiddenLayerSize + 1
        V(count,count2) = (rand(1) * initWeightInterval * 2) - initWeightInterval; % V weight vectors of output layer
    end
end

for j = 1 :  itNum
    dispText = sprintf('itNum:%d', j);
    disp(dispText);
    for k = 1 : trainSize
        for h = 1 : hiddenLayerSize
            Z(1,h+1) = 1/(1 + exp(-(W(h,:)*shufTrain(k,2:NumAtt+2)')));
        end
        Z(1,1) = 1;
        for l = 1 : K
            y(1,l) = V(l,:)*Z';
        end
        for l = 1 : K
            if l == shufTrain(k,NumAtt+3)
                ri = 1;
            else
                ri = 0;
            end
            updateV(l,1:(hiddenLayerSize + 1)) = n*(ri - y(1,l))*Z;
        end
        for h = 1 : hiddenLayerSize
            temp = 0;
            for l = 1 : K
                if l == shufTrain(k,NumAtt+3)
                    ri = 1;
                else
                    ri = 0;
                end
                temp = temp + (ri-y(1,l))*V(l,h+1);
            end
            updateW(h,1:NumAtt+1) = n*temp*Z(1,h+1)*(1- Z(1,h+1))*shufTrain(k,2:NumAtt+2);
        end
        
        V = V + updateV;
        W = W + updateW;
    end
end

%Classifying test set
dispText = sprintf('Classifying test set');
disp(dispText);
for k = 1:testSize
    for h = 1 : hiddenLayerSize
        Z(1,h+1) = 1/(1 + exp(-(W(h,:)*test(k,2:NumAtt+2)')));
    end
    Z(1,1) = 1;
    for l = 1 : K
        y(1,l) = V(l,:)*Z';
    end
    sum = 0;
    for l = 1 : K
        sum = sum + exp(y(1,l));
    end
    for l = 1 : K
        softMax(1,l) = exp(y(1,l)) / sum;
    end
    [M,I] = max(softMax');
    test(k,NumAtt+2) = I(1,1);
end
%writing to CSV file

dispText = sprintf('writing to CSV file');
disp(dispText);
for k = 1:testSize
    arr(k,1) = test(k,1);
    for l = 1:K
        if test(k,NumAtt+2) == l
            arr(k,l+1) = 1;
        else
            arr(k,l+1) = 0;
        end
    end
end

T = table(arr(:,1),arr(:,2),arr(:,3),...
    arr(:,4),arr(:,5),arr(:,6),arr(:,7),arr(:,8),arr(:,9),arr(:,10));
T.Properties.VariableNames = {'id' 'Class_1' 'Class_2' 'Class_3' 'Class_4' 'Class_5' 'Class_6' 'Class_7' 'Class_8' 'Class_9'};
writetable(T,'Submission.csv','Delimiter',',','QuoteStrings',true);











