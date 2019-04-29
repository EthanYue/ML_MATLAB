# Linear Resression

在练习中，需要实现一个单变量的线性回归。假设有一组历史数据<城市人口，开店利润>，现需要预测在哪个城市中开店利润比较好？

**总的思路为限制迭代次数，运行梯度下降函数，求出最佳拟合的Theta，便可通过Theta画出拟合后的直线；同时调用计算成本函数，求出梯度下降过程中的成本值。**

###### 假设函数(hypothesis function)
在给定一些样本数据(training set)后，采用某种学习算法(learning algorithm)对样本数据进行训练，得到了一个模型或者说是假设函数。当需要预测新数据的结果时，将新数据作为假设函数的输入，假设函数计算后得到结果，这个结果就作为预测值。

$$ h_\theta(x) = \theta^Tx $$

###### 代价函数(cost function)
学习过程就是确定假设函数的过程，或者说是：求出 θ 的过程。
现在先假设 θ 已经求出来了，就需要判断求得的这个假设函数到底好不好？它与实际值的偏差是多少？因此，就用代价函数来评估

$$ J(\theta) = \frac 1 m \sum_{n=0}^n h_\theta(x^{(i)} - y^{(i)})^2 $$

###### 梯度下降算法(Gradient descent algorithm)
梯度下降算法的本质就是求偏导数，令偏导数等于0，解出 θ

$$ \theta_j := \theta_j - \alpha \frac{\partial }{\partial \theta} J(\theta) = \theta_j - \frac \alpha m x^T(h(x^{(i)}) - y^{(i)}) = \theta_j - \frac \alpha m x^T(\theta^T x - y) $$

###### 关于分类问题和回归问题
假设函数的输出结果y（predicted y）有两种表示形式：离散的值和连续的值。比如本文中讲到的预测利润，这个结果就是属于连续的值；再比如说根据历史的天气情况预测明天的天气（下雨 or 不下雨），那预测的结果就是离散的值(discrete values)。因此，若hypothesis function输出是连续的值，则称这类学习问题为回归问题(regression problem)，若输出是离散的值，则称为分类问题(classification problem)

历史数据如下：第一列表示城市人口数，单位为万人；第二列表示利润，单位为10,000$


>```
5.5277    9.1302
8.5186   13.6620
7.0032   11.8540
.....
......
```

###### matlab文件

```
ex1.m - Octave/MATLAB script that steps you through the exercise
ex1 multi.m - Octave/MATLAB script for the later parts of the exercise
ex1data1.txt - Dataset for linear regression with one variable
ex1data2.txt - Dataset for linear regression with multiple variables
submit.m - Submission script that sends your solutions to our servers
warmUpExercise.m - Simple example function in Octave/MATLAB
plotData.m - Function to display the dataset
computeCost.m - Function to compute the cost of linear regression
gradientDescent.m - Function to run gradient descent
computeCostMulti.m - Cost function for multiple variables
gradientDescentMulti.m - Gradient descent for multiple variables
featureNormalize.m - Function to normalize features
normalEqn.m - Function to compute the normal equations
```


###### plotData代码如下
```matlab
# plotData.m
plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');

```

###### gradientDescent代码如下
```matlab
#gradientDescent.m
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
	theta = theta - (alpha/m)*X'*(theta'*X-y);
	J_history(iter) = computeCost(X, y, theta);
end
```


###### computeCost代码如下
```matlab
#computeCost.m
predictions = X * theta;
sqrErrors = (predictions-y) .^ 2;
J = 1/(2*m) * sum(sqrErrors);
```

###### 画出拟合直线
```
theta = gradientDescent(X, y, theta, alpha, iterations);
hold on; 
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
```
