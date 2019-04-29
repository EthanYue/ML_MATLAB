# Logistic Regression

用逻辑回归根据学生的考试成绩来判断该学生是否可以入学。


**总的思路为限制迭代次数，运行梯度下降函数，求出最佳拟合的Theta，便可通过Theta画出拟合后的直线；同时调用计算成本函数，求出梯度下降过程中的成本值。**

###### sigmoid function
对于 logistic regression而言，它针对的是 classification problem。这里只讨论二分类问题，比如上面的“根据成绩入学”，结果只有两种：y==0时，成绩未合格，不予入学；y==1时，可入学。即，y的输出要么是0，要么是1
如果采用 linear regression，它的假设函数是这样的：

$$ h_\theta(x) = \theta^Tx $$

假设函数的取值即可以远远大于1，也可以远远小于0，并且容易受到一些特殊样本的影响。比如在上图中，就只能约定：当假设函数大于等于0.5时；预测y==1，小于0.5时，预测y==0。

而如果引入了sigmoid function，就可以把假设函数的值域“约束”在[0, 1]之间。总之，引入sigmoid function，就能够更好的拟合分类问题中的数据，即从这个角度看：regression model 比 linear model 更合适 classification problem.
引入sigmoid后，假设函数如下

$$ g(z) = \frac {1} {1 + e^{-z}}  $$

所以最终假设函数为：

$$ h_\theta(x) = \frac {1} {1 + e^{-\theta^Tx}} $$

###### 代价函数(cost function)
学习过程就是确定假设函数的过程，或者说是：求出 θ 的过程。
现在先假设 θ 已经求出来了，就需要判断求得的这个假设函数到底好不好？它与实际值的偏差是多少？因此，就用代价函数来评估

$$ J(\theta) = -\frac 1 m [log(g(\theta^Tx^T))y + log(1 - g(\theta^Tx^T))(1 - y)] $$

###### 梯度下降算法(Gradient descent algorithm)
梯度下降算法，本质上是求导数(偏导数)，或者说是：方向导数。方向导数所代表的方向--梯度方向，下降得最快。
而我们知道，对于某些图形所代表的函数，它可能有很多个导数为0的点，这类函数称为非凸函数(non-convex function)；而某些函数，它只有一个全局唯一的导数为0的点，称为 convex function
convex function能够很好地让Gradient descent寻找全局最小值。而上图左边的non-convex就不太适用Gradient descent了。
就是因为上面这个原因，logistic regression 的 cost function被改写成了下面这个公式

$$ \frac{\partial  J(\theta)}{\partial \theta} = \frac 1m [x^T g(x\theta ) - y)] $$

###### 逻辑回归的正则化（Regularized logistic regression）
正则化就是为了解决过拟合问题(overfitting problem)。
一般而言，当模型的特征(feature variables)非常多，而训练的样本数目(training set)又比较少的时候，训练得到的假设函数(hypothesis function)能够非常好地匹配training set中的数据，此时的代价函数几乎为0。下图中最右边的那个模型 就是一个过拟合的模型。
正是因为 feature variable非常多，导致 hypothesis function 的幂次很高，hypothesis function变得很复杂，最终拟合的曲线方程为：

$$ y = \theta_0 + \theta_1x + \theta_2 x^2 + \theta_3 x^3 + \theta_4 x^4 $$

如果添加一个"正则化项"，减少 高幂次的特征变量的影响，那 hypothesis function就会变得平滑，正如前面提到，梯度下降算法的目标是最小化cost function，而现在把 theta(3) 和 theta(4)的系数设置为1000，设得很大，求偏导数时，相应地得到的theta(3) 和 theta(4) 就都约等于0了。
更一般地，我们对每一个theta(j)，j>=1，进行正则化，就得到了一个如下的代价函数：其中的 lambda(λ)就称为正则化参数(regularization parameter)；所以正则化后的成本函数为：

$$ J(\theta) = -\frac {1} {2m} [log(g(\theta^Tx^T))y + log(1 - g(\theta^Tx^T))(1 - y) + \lambda \sum_{i=1}^n \theta_j ^2] $$

从上面的J(theta)可以看出：如果lambda(λ)=0，则表示没有使用正则化；如果lambda(λ)过大，使得模型的各个参数都变得很小，导致h(x)=theta(0)，从而造成欠拟合；如果lambda(λ)很小，则未充分起到正则化的效果。因此，lambda(λ)的值要合适。正则化后的梯度公式如下：

$$ \frac{\partial  J(\theta)}{\partial \theta} = \frac 1m [x^T g(x\theta ) - y)]\ (for j = 1)$$

$$ \frac{\partial  J(\theta)}{\partial \theta} = \frac 1m [x^T g(x\theta ) - y)] + \frac \lambda m \theta_j \ (for j \ge 1)$$

>这里的训练数据(training instance)是学生的两次考试成绩，以及TA是否能够入学的决定（y=0表示成绩不合格，不予录取；y=1表示录取）因此，需要根据trainging set 训练出一个classification model。然后，拿着这个classification model 来评估新学生能否入学。

>训练数据的成绩样例如下：第一列表示第一次考试成绩，第二列表示第二次考试成绩，第三列表示入学结果（0--不能入学，1--可以入学）

>历史数据如下：第一列表示城市人口数，单位为万人；第二列表示利润，单位为10,000$
```
34.62365962451697, 78.0246928153624,  0
30.28671076822607, 43.89499752400101, 0
35.84740876993872, 72.90219802708364, 0
60.18259938620976, 86.30855209546826, 1
....
....
....
```

通过调用costfunction.m文件中定义的coustFunction函数，从而运行梯度下降算法找到使代价函数J(theta)最小化的 逻辑回归模型参数theta。调用costFunction函数的代码如下：
```
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```
从上面代码的最后一行可以看出，我们是通过 fminunc 调用 costFunction函数，来求得 theta的，而不是自己使用 Gradient descent 在for 循环求导来计算 theta。
通过Gradient descent算法求得了theta，将theta代入到假设函数中，就得到了 logistic regression model。
###### matlab文件
```
ex2.m - Octave/MATLAB script that steps you through the exercise
ex2 reg.m - Octave/MATLAB script for the later parts of the exercise
ex2data1.txt - Training set for the rst half of the exercise
ex2data2.txt - Training set for the second half of the exercise
submit.m - Submission script that sends your solutions to our servers
mapFeature.m - Function to generate polynomial features
plotDecisionBoundary.m - Function to plot classier's decision bound-
ary
plotData.m - Function to plot 2D classication data
sigmoid.m - Sigmoid Function
costFunction.m - Logistic Regression Cost Function
predict.m - Logistic Regression Prediction Function
costFunctionReg.m - Regularized Logistic Regression Cost
```

###### plotData代码如下
```matlab
# plotData.m
pos = find(y==1);
neg = find(y==0);
plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
```

###### sigmoid代码如下
```matlab
# sigmoid.m
g = 1./(ones(size(z)) + exp(-z)); % ‘点除’ 表示 1 除以矩阵(向量)中的每一个元素
```

###### costFunction代码如下
```matlab
#costFunction.m
m = length(y);
J = 0;
grad = zeros(size(theta));
J = (log(sigmoid(theta'*X')) * y + log(1-sigmoid(theta'*X')) * (1 - y))/(-m);
grad = (X' * (sigmoid(X*theta)-y))/m;
```

###### costFunctionReg代码如下
```matlab
#costFunctionReg.m
m = length(y);
J = 0;
grad = zeros(size(theta));
J = ( log( sigmoid(theta'*X') ) * y + log( 1-sigmoid(theta'*X') ) * (1 - y) )/(-m) + (lambda / (2*m)) * ( ( theta( 2:length(theta) ) )' * theta(2:length(theta)) );
grad = ( X' * ( sigmoid(X*theta)-y ) )/m + ( lambda / m ) * ( [0; ones( length(theta) - 1 , 1 )].*theta );
```

###### 调用costFunctionReg.m的代码如下：
```
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
% Plot Boundary
plotDecisionBoundary(theta, X, y);
```