# Multi-Class & Neural Networks(feedforward propagation)

使用逻辑回归多分类和神经网络(neural networks)识别手写的阿拉伯数字(0-9)。

## Multi-Class
#####使用逻辑回归来实现多分类问题(one-vs-all)
所谓多分类问题，是指分类的结果为三类以上。比如，预测明天的天气结果为三类：晴(用y==1表示)、阴(用y==2表示)、雨(用y==3表示)

分类的思想，其实与逻辑回归分类(默认是指二分类，binary classification)很相似，对“晴天”进行分类时，将另外两类(阴天和下雨)视为一类：(非晴天)，这样，就把一个多分类问题转化成了二分类问题。示意图如下：（图中的圆圈 表示：不属于某一类的 所有其他类）

对于N分类问题(N>=3)，就需要N个假设函数(预测模型)，也即需要N组模型参数θ（θ一般是一个向量）

然后，对于每个样本实例，依次使用每个模型预测输出，选取输出值最大的那组模型所对应的预测结果作为最终结果。

因为模型的输出值，在sigmoid函数作用下，其实是一个概率值。，注意：hθ(1)(x)，hθ(2)(x)，hθ(3)(x)三组 模型参数θ 一般是不同的。比如：
>hθ(1)(x)，输出 预测为晴天(y==1)的概率
hθ(2)(x)，输出 预测为阴天(y==2)的概率
hθ(3)(x)，输出 预测为雨天(y==3)的概率

对于上面的识别阿拉伯数字的问题，一共需要训练出10个逻辑回归模型，每个逻辑回归模型对应着识别其中一个数字。

我们一共有5000个样本，样本的预测结果值就是：y=(1,2,3,4,5,6,7,8,9,10)，其中 10 代表 数字0

我们使用Matlab fmincg库函数 来求解使得代价函数取最小值的 模型参数θ


##### matlab文件
```
ex3.m - Octave/MATLAB script that steps you through part 1
ex3 nn.m - Octave/MATLAB script that steps you through part 2
ex3data1.mat - Training set of hand-written digits
ex3weights.mat - Initial weights for the neural network exercise
submit.m - Submission script that sends your solutions to our servers
displayData.m - Function to help visualize the dataset
fmincg.m - Function minimization routine (similar to fminunc)
sigmoid.m - Sigmoid function
lrCostFunction.m - Logistic regression cost function
oneVsAll.m - Train a one-vs-all multi-class classifier
predictOneVsAll.m - Predict using a one-vs-all multi-class classifier
predict.m - Neural network prediction function
```

##### oneVsall代码如下
```matlab
# oneVsall.m
m = size(X, 1);% num of samples
n = size(X, 2);% num of features
% You need to return the following variables correctly
all_theta = zeros(num_labels, n + 1);
% Add ones to the X data matrix
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj','on','MaxIter',50);
for c = 1:num_labels
all_theta(c, :) = fmincg(@(t)(lrCostFunction(t, X, (y == c),lambda)), initial_theta,options );
end
```

下面解释一下 for循环：
num_labels 为分类器个数，共10个，每个分类器(模型)用来识别10个数字中的某一个。

我们一共有5000个样本，每个样本有400中特征变量，因此：模型参数θ 向量有401个元素。

initial_theta = zeros(n + 1, 1); % 模型参数θ的初始值(n == 400)

all_theta是一个10*401的矩阵，每一行存储着一个分类器(模型)的模型参数θ 向量，执行上面for循环，就调用fmincg库函数 求出了 所有模型的参数θ 向量了。

求出了每个模型的参数向量θ，就可以用 训练好的模型来识别数字了。对于一个给定的数字输入(400个 feature variables) input instance，每个模型的假设函数hθ(i)(x) 输出一个值(i = 1,2,...10)。取这10个值中最大值那个值，作为最终的识别结果。比如g(hθ(8)(x))==0.96 比其它所有的 g(hθ(i)(x)) (i = 1,2,...10,但 i 不等于8) 都大，则识别的结果为 数字 8

##### lrcostFunction代码如下
```matlab
#lrcostFunction.m
m = length(y);
J = 0;
grad = zeros(size(theta));
J = ( log( sigmoid(theta'*X') ) * y + log( 1-sigmoid(theta'*X') ) * (1 - y) )/(-m) + (lambda / (2*m)) * ( ( theta( 2:length(theta) ) )' * theta(2:length(theta)) );
grad = ( X' * ( sigmoid(X*theta)-y ) )/m + ( lambda / m ) * ( [0; ones( length(theta) - 1 , 1 )].*theta );
```

##### predictOneVsAll代码如下
```matlab
#predictOneVsAll.m
[~,p] = max( X * all_theta',[],2); % 求矩阵(X*all_theta')每行的最大值，p 记录矩阵每行的最大值的索引
```

##### 运行逻辑回归多分类代码如下：
```
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

pred = predictOneVsAll(all_theta, X);
```
---

## Neural Networks(feedforward propagation)
由于逻辑回归是线性分类（它的假设函数是一个线性函数，就是划一条直线，把数据分成了两类。可参考这篇文章中的：②使用逻辑回归来实现多分类问题(one-vs-all) 部分 的图片）

对于一些复杂的类别，逻辑回归就解决不了了。(无法通过 划直线 将 不同类别 分开）


而神经网络，则能够实现很复杂的非线性分类问题。

对于神经网络而言，同样有一个训练样本矩阵 X，同时还有一个模型参数 Theta 矩阵，通过某种算法将模型参数矩阵训练好之后(求出 Theta 矩阵)，再使用前向传播算法( feedforward propagation algorithm)（感觉就像是矩阵相乘），就可以对输入的测试样本进行预测了。

本作业中， 模型参数 Theta 矩阵是已经训练好了的，直接 load 到Matlab中即可。

##### predict代码如下
```matlab
#predict.m
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);% p 是 5000*1向量

% 模拟实现前向传播算法
X = [ones(m, 1) X];
a_super_2 = sigmoid(Theta1 * X');
a_super_2 = [ones(1,m); a_super_2];% add bias unit
a_super_3 = sigmoid(Theta2 * a_super_2);

[~,p] = max( a_super_3' ,[], 2 ); % 对样本的结果进行预测，与逻辑回归的预测类似，选取输出的最大值 作为最终的预测结果
```
我们正是通过Matlab 的 max 函数，求得矩阵 a_super3′ 的每一行的最大值。将每一行的中的最大值 的索引 赋值给向量p。其中，a_super3′ 是一个5000行乘10列的矩阵 
向量p就是预测的结果向量。而由于 a_super3′ 有10列，故 p 中每个元素的取值范围为[1,10]，即分别代表了数字 0-9

##### 运行神经网络正向传播代码如下：
```
m = size(X, 1);
rp = randperm(m);
for i = 1:m
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));
    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end
```