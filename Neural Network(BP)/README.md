# Neural Networks(back propagation)

实现一个BP(backpropagation)算法，并将之应用到手写的阿拉伯数字(0-9)的自动识别上。

>训练数据集(training set)如下：一共有5000个训练实例(training instance)，每个训练实例是一个400维特征的列向量(20*20 pixel image)。用 X 矩阵表示整个训练集，则 X 是一个 5000*400 （5000行 400列）的矩阵
>另外，还有一个5000*1的列向量 y ，用来标记训练数据集的结果。比如，第一个训练实例对应的输出结果为数字：5

##### 模型表示
我们使用三层的神经网络模型：输入层、一个隐藏层、和输出层。将训练数据集矩阵 X 中的每一个训练实例 用load指令加载到Matlab中

由于我们使用三层的神经网络，故一共有二个参数矩阵：Θ(1) (Theta1)和 Θ(2) (Theta2)，它是预先存储在 ex4weights.mat文件中，使用load('ex4weights')加载到Matlab中

参数矩阵Θ(1) (Theta1) 和 Θ(2) (Theta2) 的维数是如何确定的呢？

一般，对于一个特定的训练数据集而言，它的输入维数 和输出的结果数目是确定的。比如，本文中的数字图片是用一个400维的特征向量代表，输出结果则是0-9的阿拉伯数字，即一共有10种输出。而中间隐藏层的个数则是变化的，根据实际情况选择隐藏层的数目。本文中隐藏层单元个数为25个。故参数矩阵Θ(1) (Theta1)是一个25 x 401矩阵，行的数目为25，由隐藏层的单元个数决定(不包括bias unit)，列的数目由输入特征数目决定(加上bias unit之后变成401)。同理，参数矩阵Θ(2) (Theta2)是一个10 x 26矩阵，行的数目由输出结果数目决定（0-9 共10种数字），列的数目由隐藏层数目决定（25个隐藏层的unit，再加上一个bias unit）。

##### 代价函数
未考虑正则化的神经网络的代价函数如下：

$$ J(\theta)=\frac 1 m \sum_{i=1}^{m} \sum_{k=1}^{K} [-y_k^{(i)}log(h_\theta(x^{(i)})_k) - (1-y_k)^{(i)}log(1-(h_\theta(x^{(i)})_k))]$$

其中，m等于5000，表示一共有5000个训练实例；K=10，总共有10种可能的训练结果（数字0-9）

#####假设函数 hθ(x(i)) 和 hθ(x(i))k 的解释

###### 我们是通过如下公式来求解hθ(x(i))的：

* a(1) = x  再加上bias unit a0(1) ，其中，x 是训练集矩阵 X 的第 i 行（第 i 个训练实例）。它是一个400行乘1列的 列向量，上标(1)表示神经网络的第几层。

* z(2) =  Θ(1) *  a(1)。

* a(2) = sigmoid(z(2)) 使用 sigmoid函数作用于z(2)，就得到了a(2)，它代表隐藏层的每个神经元的值。a(2)是一个25行1列的列向量，再将隐藏层的25个神经元，添加一个bias unit ，就a0(2)可以计算第三层（输出层）的神经单元向量a(3)了。a(3)是一个10行1列的列向量。

* 同理，z(3) =  Θ(2) *  a(2)

* a(3) = sigmoid(z(3)) 此时得到的 a(3) 就是假设函数 hθ(x(i))

由此可以看出：假设函数 hθ(x(i))就是一个10行1列的列向量，而 hθ(x(i))k 就表示列向量中的第 k 个元素。【也即该训练实例以 hθ(x(i))k 的概率取数字k？】

举个例子： hθ(x(6)) = (0, 0, 0.03, 0, 0.97, 0, 0, 0, 0, 0)T

它是含义是：使用神经网络 训练 training set 中的第6个训练实例，得到的训练结果是：以0.03的概率是 数字3，以0.97的概率是 数字5

（注意：向量的下标10 表示 数字0）

###### 训练样本集的 结果向量 y (label of result)的解释

由于神经网络的训练是监督学习，也就是说：样本训练数据集是这样的格式：(x(i), y(i))，对于一个训练实例x(i)，我们是已经确定知道了它的正确结果是y(i)，而我们的目标是构造一个神经网络模型，训练出来的这个模型的假设函数hθ(x)，对于未知的输入数据x(k)，能够准确地识别出正确的结果。

因此，训练数据集(traing set)中的结果数据 y 是正确的已知的结果，比如y(600) = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0)T 表示：训练数据集中的第600条训练实例它所对应的正确结果是：数字5 （因为，向量y(600)中的第5个元素为1，其它所有元素为0），另外需要注意的是：当向量y(i) 第10个元素为1时 代表数字0。

##### BP(back propagation)算法

BP算法是用来 计算神经网络的代价函数的梯度。

计算梯度，本质上是求偏导数。来求解偏导数我们可以用传统的数学方法：求偏导数的数学计算公式来求解。参数矩阵Θ非常大时，就需要大量地进行计算了，而BP算法，则解决了这个计算量过大的问题。BP算法又称反向传播算法，它从输出层开始，往输入层方向以“某种形式”的计算，得到一组“数据“，而这组数据刚好就是我们所需要的 梯度。

##### Sigmoid 函数的导数

至于为什么要引入Sigmoid函数，Sigmoid函数的导数有一个特点，即Sigmoid的导数可以用Sigmoid函数自己本身来表示，如下：

$$ g(z) = sigmoid(z) = \frac {1} {1 + e^{-z}} $$

$$ g^`(z) = g(z) (1 - g(z)) $$

##### 训练神经网络时的“symmetry”现象---随机初始化神经网络的参数矩阵(权值矩阵Θ)

随机初始化参数矩阵，就是对参数矩阵Θ(L)中的每个元素，随机地赋值，取值范围一般为[ξ ,-ξ]。

假设将参数矩阵Θ(L) 中所有的元素初始化0，则 根据计算公式：a1(2) = Θ(1) * a(2) 中的每个元素都会取相同的值。

因此，随机初始化的好处就是：让学习 更有效率

随机初始化的Matlab实现如下：可以看出，它是先调用 randInitializeWeights.m 中定义的公式进行初始化的。然后，再将 initial_Theta1 和 initial_Theta2 unroll 成列向量

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

##### BP算法的具体执行步骤如下：

对于每一个训练实例(x, y)，先用“前向传播算法”计算出 activations（a(2)， a(3)），然后再对每一层计算一个残差δj(L) (error term)。注意：输入层(input layer)不需要计算残差。

###### 具体的每一层的残差计算公式如下：（本文中的神经网络只有3层，隐藏层的数目为1）

* 对于输出层的残差计算公式如下，这里的输入层是第1层，隐藏层是第2层，输出层是第3层

$$ \delta_k^{(3)} = a_k^{(3)} - y^{(k)} $$

>残差δ(3)是一个向量，下标 k 表示，该向量中的第 k 个元素。yk 就是前面提到的表示 样本结果的向量y 中的第 k 个元素。(这里的向量 y 是由训练样本集的结果向量y分割得到的)

>上面的减法公式用向量表示为：δ(3)= a(3) - y，因此δ(3)维数与a(3)一样，它与分类输出的数目有关。在本例中，是10，故δ(3)是一个10*1向量

* 对于隐藏层的残差计算公式如下：

$$ \delta_k^{(2)} = (\theta^{(2)})^T\delta_k^{(3)} .* g^`(z^{(2)}) $$

当每一层的残差计算好之后，就可以更新 Δ(delta) 矩阵了，Δ(delta) 矩阵与 参数矩阵有相同的维数，初始时Δ(delta) 矩阵中的元素全为0.

它的定义(计算公式)如下：

$$ \Delta^{(l)} = \Delta^{(l)}+\delta_k^{(l+1)} (a^{(l)})^T $$

在这里，δ(L+1)是一个列向量，(a(1))T是一个行向量，相乘后，得到的是一个矩阵。
计算出 Δ(delta) 矩阵后，就可以用它来更新 代价函数的导数了，公式如下：

$$ \frac {\partial } {\partial \Theta_{ij} ^{(l)}} J(\Theta) = D_{ij} ^{(l)} = \frac 1m \Delta_{ij} ^{(l)}$$

##### 梯度检查(gradient checking)

梯度检查的原理如下：由于我们通过BP算法这种巧妙的方式求得了代价函数的导数，那它到底正不正确呢？这里就可以用 高等数学 里面的导数的定义(极限的定义)来计算导数，然后再比较：用BP算法求得的导数 和 用导数的定义 求得的导数 这二者之间的差距。

导数定义(极限定义)---非正式定义，如下：

$$ f_i(\theta) \approx \frac {J(\theta^{(i+\epsilon)}) - J(\theta^{(i-\epsilon)})} {2\epsilon}  $$

正是这种通过定义直接计算的方式 运算量很大，在正式训练时，要记得关闭 gradient checking

如果gradient checking 结果中二者计算出来的结果几乎相等，则 BP算法的运行是正常的。

##### 神经网络的正则化

对于神经网络而言，它的表达能力很强，容易出现 overfitting problem，故一般需要正则化。正则化就是加上一个正则化项，就可以了。注意 bias unit不需要正则化

$$ \frac {\partial } {\partial \Theta_{ij} ^{(l)}} J(\Theta) = D_{ij} ^{(l)} = \frac 1m \Delta_{ij} ^{(l)} \ (for j=0) $$

$$  \frac {\partial } {\partial \Theta_{ij} ^{(l)}} J(\Theta) = D_{ij} ^{(l)} = \frac 1m \Delta_{ij} ^{(l)} + \frac \lambda m \Theta_{ij} ^{(l)}(for j\ge1) $$

##### 使用Matlab的 fmincg 函数 最终得到 参数矩阵Θ
```
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
```

##### sigmoidGradient代码如下
```matlab
# sigmoidGradient.m
g = sigmoid(z) .* (1 - sigmoid(z));
```

##### nnCostFunction代码如下
```matlab
# nnCostFunction.m
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));% Theta1_grad is a 25*401 matrix
Theta2_grad = zeros(size(Theta2));% Theta2_grad is a 10*26 matrix


% Feedforward the neural network
X = [ones(m,1) X]; %5000*401
a_super2 = sigmoid(Theta1 * X'); % attention a_super2 is a 25*5000 matrix
a_super2 = [ones(1,m);a_super2]; %add each bias unit for a_superscript2, 26 * 5000
a_super3 = sigmoid(Theta2 * a_super2);%10*5000
a3 = 1 - a_super3;%10*5000

%将5000条的结果label 向量y 转化成元素只为0或1 的矩阵Y
Y = zeros(num_labels, m); %10*5000, each column is a label result
for i = 1:num_labels
    Y(i, y==i)=1;
end

Y1 = 1 - Y;
res1 = 0;
res2 = 0;
for j = 1:m
    %两个矩阵的每一列相乘,再把结果求和。预测值和结果label对应的元素相乘,就是某个输入x 的代价
    tmp1 = sum( log(a_super3(:,j)) .* Y(:,j) );
    res1 = res1 + tmp1; % m 列之和
    tmp2 = sum( log(a3(:,j)) .* Y1(:,j) );
    res2 = res2 + tmp2;
end
J = (-res1 - res2) / m;


% the backpropagation algorithm
for i = 1:m
    a1 = X(i, :)'; %the i th input variables, 400*1
    z2 = Theta1 * a1;
    a2 =  sigmoid( z2 ); % Theta1 * x superscript i
    a2 = [ 1; a2 ];% add bias unit, a2's size is 26 * 1
    z3 = Theta2 * a2;
    a3 = sigmoid( z3 ); % h_theta(x)
    error_3 = a3 - Y( :, i ); % last layer's error, 10*1
    err_2 =  Theta2' * error_3; % 26*1
    error_2 = ( err_2(2:end) ) .*  sigmoidGradient(z2);% 去掉 bias unit 对应的 error units
    Theta2_grad = Theta2_grad + error_3 * a2';
    Theta1_grad = Theta1_grad + error_2 * a1';
end

Theta2_grad = Theta2_grad / m; % video 9-2 backpropagation algorithm the 11 th minute
Theta1_grad = Theta1_grad / m;

% regularization with the cost function and gradients.
% reg for cost function J,  ex4.pdf page 6
Theta1_tmp = Theta1(:, 2:end).^2;
Theta2_tmp = Theta2(:, 2:end).^2;
reg = lambda / (2*m) * ( sum( Theta1_tmp(:) ) + sum( Theta2_tmp(:) ) );
J = (-res1 - res2) / m + reg;
% reg for bp, ex4.pdf materials page 11
Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = Theta1_grad + lambda / m * Theta1;
Theta2_grad = Theta2_grad + lambda / m * Theta2;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
```

##### 运行神经网络代码如下：
```
load('ex4weights.mat');
% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];
%% ================ Part 3: Compute Cost (Feedforward) ================
lambda = 0;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

%% =============== Part 4: Implement Regularization ===============
lambda = 1;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

%% ================ Part 5: Sigmoid Gradient  ================
g = sigmoidGradient([-1 -0.5 0 0.5 1]);

%% ================ Part 6: Initializing Pameters ================
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =============== Part 7: Implement Backpropagation ===============
%  Check gradients by running checkNNGradients
checkNNGradients;

%% =============== Part 8: Implement Regularization ===============
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

%% =================== Part 8: Training NN ===================
options = optimset('MaxIter', 50);
lambda = 1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% ================= Part 9: Visualize Weights =================
displayData(Theta1(:, 2:end));

%% ================= Part 10: Implement Predict =================
pred = predict(Theta1, Theta2, X);
```