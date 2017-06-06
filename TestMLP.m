%Berke Oral 150130127
%This file is to test & optimize MLP classifier on test data sample
%
%Reads from modified csv file where first row removed and class values
%changed to numbers (class3 to 3)

clear;
train = load('train.csv');
%test = load('test.csv');
%testSize = size(test, 1);
shufTrain = train(randperm(size(train,1)),:);

trainSize = size(shufTrain, 1);
shufTrain = [shufTrain(:,1) ones(trainSize,1) shufTrain(:, 2:95)];
NumAtt = size(shufTrain, 2) - 3;


hiddenLayerSize = 64;
n = 0.20;
itNum = 5;
K = 9;

initWeightInterval = 0.01;
TestResults = [0 0 0 0 0 0];


for hiddenLayerSize = 64 : 8 : 64
    for itNum = [5 10]
        for n = [0.005 0.025 0.05 0.1 0.25]
            for initWeightInterval = [0.001 0.005 0.01 0.05 0.1 0.25]
                dispText = sprintf('H: %d, itNum: %d, n: %d, initW: %d', hiddenLayerSize,itNum,n,initWeightInterval);
                disp(dispText);
                TimerStart = tic;
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
                
                %trains for first 90% of data than tests accuracy on last %10
                for j = 1 :  itNum
                    for k = 1 : (trainSize * 9/10)
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
   
                %---Test
                totalClassification = 0;
                correctlyClassified = 0;
                
                currentK = k;
                
                for k = currentK:trainSize
                    for h = 1 : hiddenLayerSize
                        Z(1,h+1) = 1/(1 + exp(-(W(h,:)*shufTrain(k,2:NumAtt+2)')));
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
                    totalClassification = totalClassification +1;
                    if I(1,1) == shufTrain(k, NumAtt + 3)
                        correctlyClassified = correctlyClassified +1;
                    end
                end
                TimePassed = toc(TimerStart);
                
                testAccuracy = 100* correctlyClassified / totalClassification;
                TestResults(end + 1, 1 : 6) = [hiddenLayerSize itNum n initWeightInterval testAccuracy TimePassed];
                dispText = sprintf('Acc: %d, Time Passed: %d',testAccuracy, TimePassed);
                disp(dispText);
            end
        end
    end
end

TestResults(1,:) = [];








