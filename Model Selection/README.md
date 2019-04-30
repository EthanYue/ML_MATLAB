# Debugging a learning algorithm

根据水库中蓄水标线(water level) 使用正则化的线性回归模型预 水流量(water flowing out of dam)，然后 debug 学习算法 以及 讨论偏差和方差对 该线性回归模型的影响。

>本作业的数据集分成三部分：
* 训练集(training set)，样本矩阵(训练集)：X，结果标签(label of result)向量 y
* 交叉验证集(cross validation set)，确定正则化参数 Xval 和 yval
* 测试集(test set) for evaluating performance，测试集中的数据 是从未出现在 训练集中的

>训练集中一共有12个训练实例，每个训练实例只有一个特征。

##### 调试机器学习算法的方法
* 寻找更多的训练数据
* 减少训练集的特征数量
* 增加训练集额外的特征
* 增加多项式的次数
* 增大λ
* 减小λ

##### 训练集、验证集、测试集
调试模型的过程中，为了验证不同训练模型的准确性，应该对训练后的模型进行测试，但如果直接在测试集上测试不同的模型，则最终损失最小的模型将只拟合于某一特定的测试集，所以需要将训练数据分成训练集、验证集、测试集三部分，比例为60%，20%，20%，将训练后的模型在验证集上进行验证，以此挑选出的模型将更具泛化性。

##### 诊断高偏差（Bias）和高方差（Variance）
高偏差即为欠拟合（Underfitting）；高方差即为过拟合（Overfitting），高偏差和高方差的主要表现：
* 多项式特征的最高阶层（d）

$$ h(\theta) = \theta_0 + \theta_1x + \theta_2x^2 + \theta_nx^d $$

$$ J(\theta) = \frac 1 {2m} \sum_{i=1}^n (h(x^{(i)}) - y^{(i)}) ^2$$
* 
 * d过大或者过小，都会导致训练集和验证集的损失增大；但d过小，训练集和验证集的损失差越接近，d过大，训练集和验证集的损失差越大

* 正则化的参数 λ

$$ Reg(\theta) = \frac {\lambda} {2m} \sum_{j=1}^m \theta_j^2$$

* 
 * λ 过大会导致梯度下降后，theta过小，从而多项式的d变小，出现高偏差；λ 过大会导致梯度下降后，theta过大，从而多项式的d变大，出现高方差。
 * λ 过大或者过小，都会导致训练集和验证集的损失增大；但λ过小，训练集和验证集的损失差越大，λ过大，训练集和验证集的损失差越接近。

* 学习曲线，训练集的数量 m
 * 高偏差情况下，初期 m 增大，训练集和验证集的损失会减小，但减小到一定程度后，训练集和验证集的损失将会趋于平缓，不会发生变化，所以此时增大训练集的数量无效。
  * 高方差情况下，初期 m 增大，训练集和验证集的损失会减小，但减小到一定程度后， 训练集和验证集的损失将会出现一段较大的差距，但随着训练集数量的增大，该差距会越来越小

##### 调试方法总结
* 适用于高偏差情况
 * 增加训练集的特征
 * 增加拟合多项式的阶层
 * 减小 λ
* 适用于高方差情况
 * 减小训练集的特征
 * 减小拟合多项式的阶层
 * 增大 λ


##### linearRegCostFunction代码如下
```matlab
# linearRegCostFunction.m
reg = (lambda/(2*m)) * ((theta(2:length(theta)))' * theta(2:length(theta)));
J = sum((X * theta- y).^ 2) / (2*m) + reg;


grad_tmp = X' * (X*theta - y) ./ m;
grad = [grad_tmp(1:1); grad_tmp(2: end) + (lambda/m) * theta(2:end)];
```

##### learningCurve代码如下
```matlab
# learningCurve.m
for i = 1:m
        theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
        error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
        error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end
```

##### polyFeatures代码如下
```matlab
# polyFeatures.m
for i = 1:p
    X_poly(:,i) = X.^i;
end
```

##### validationCurve代码如下
```matlab
# validationCurve.m
for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    theta = trainLinearReg(X, y, lambda);
    error_train(i) = linearRegCostFunction(X, y, theta, 0);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end
```