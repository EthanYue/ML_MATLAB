# Neural Networks(back propagation)

ʵ��һ��BP(backpropagation)�㷨������֮Ӧ�õ���д�İ���������(0-9)���Զ�ʶ���ϡ�

>ѵ�����ݼ�(training set)���£�һ����5000��ѵ��ʵ��(training instance)��ÿ��ѵ��ʵ����һ��400ά������������(20*20 pixel image)���� X �����ʾ����ѵ�������� X ��һ�� 5000*400 ��5000�� 400�У��ľ���
>���⣬����һ��5000*1�������� y ���������ѵ�����ݼ��Ľ�������磬��һ��ѵ��ʵ����Ӧ��������Ϊ���֣�5

##### ģ�ͱ�ʾ
����ʹ�������������ģ�ͣ�����㡢һ�����ز㡢������㡣��ѵ�����ݼ����� X �е�ÿһ��ѵ��ʵ�� ��loadָ����ص�Matlab��

��������ʹ������������磬��һ���ж����������󣺦�(1) (Theta1)�� ��(2) (Theta2)������Ԥ�ȴ洢�� ex4weights.mat�ļ��У�ʹ��load('ex4weights')���ص�Matlab��

��������(1) (Theta1) �� ��(2) (Theta2) ��ά�������ȷ�����أ�

һ�㣬����һ���ض���ѵ�����ݼ����ԣ���������ά�� ������Ľ����Ŀ��ȷ���ġ����磬�����е�����ͼƬ����һ��400ά������������������������0-9�İ��������֣���һ����10����������м����ز�ĸ������Ǳ仯�ģ�����ʵ�����ѡ�����ز����Ŀ�����������ز㵥Ԫ����Ϊ25�����ʲ�������(1) (Theta1)��һ��25 x 401�����е���ĿΪ25�������ز�ĵ�Ԫ��������(������bias unit)���е���Ŀ������������Ŀ����(����bias unit֮����401)��ͬ����������(2) (Theta2)��һ��10 x 26�����е���Ŀ����������Ŀ������0-9 ��10�����֣����е���Ŀ�����ز���Ŀ������25�����ز��unit���ټ���һ��bias unit����

##### ���ۺ���
δ�������򻯵�������Ĵ��ۺ������£�

$$ J(\theta)=\frac 1 m \sum_{i=1}^{m} \sum_{k=1}^{K} [-y_k^{(i)}log(h_\theta(x^{(i)})_k) - (1-y_k)^{(i)}log(1-(h_\theta(x^{(i)})_k))]$$

���У�m����5000����ʾһ����5000��ѵ��ʵ����K=10���ܹ���10�ֿ��ܵ�ѵ�����������0-9��

#####���躯�� h��(x(i)) �� h��(x(i))k �Ľ���

###### ������ͨ�����¹�ʽ�����h��(x(i))�ģ�

* a(1) = x  �ټ���bias unit a0(1) �����У�x ��ѵ�������� X �ĵ� i �У��� i ��ѵ��ʵ����������һ��400�г�1�е� ���������ϱ�(1)��ʾ������ĵڼ��㡣

* z(2) =  ��(1) *  a(1)��

* a(2) = sigmoid(z(2)) ʹ�� sigmoid����������z(2)���͵õ���a(2)�����������ز��ÿ����Ԫ��ֵ��a(2)��һ��25��1�е����������ٽ����ز��25����Ԫ�����һ��bias unit ����a0(2)���Լ�������㣨����㣩���񾭵�Ԫ����a(3)�ˡ�a(3)��һ��10��1�е���������

* ͬ��z(3) =  ��(2) *  a(2)

* a(3) = sigmoid(z(3)) ��ʱ�õ��� a(3) ���Ǽ��躯�� h��(x(i))

�ɴ˿��Կ��������躯�� h��(x(i))����һ��10��1�е����������� h��(x(i))k �ͱ�ʾ�������еĵ� k ��Ԫ�ء���Ҳ����ѵ��ʵ���� h��(x(i))k �ĸ���ȡ����k����

�ٸ����ӣ� h��(x(6)) = (0, 0, 0.03, 0, 0.97, 0, 0, 0, 0, 0)T

���Ǻ����ǣ�ʹ�������� ѵ�� training set �еĵ�6��ѵ��ʵ�����õ���ѵ������ǣ���0.03�ĸ����� ����3����0.97�ĸ����� ����5

��ע�⣺�������±�10 ��ʾ ����0��

###### ѵ���������� ������� y (label of result)�Ľ���

�����������ѵ���Ǽලѧϰ��Ҳ����˵������ѵ�����ݼ��������ĸ�ʽ��(x(i), y(i))������һ��ѵ��ʵ��x(i)���������Ѿ�ȷ��֪����������ȷ�����y(i)�������ǵ�Ŀ���ǹ���һ��������ģ�ͣ�ѵ�����������ģ�͵ļ��躯��h��(x)������δ֪����������x(k)���ܹ�׼ȷ��ʶ�����ȷ�Ľ����

��ˣ�ѵ�����ݼ�(traing set)�еĽ������ y ����ȷ����֪�Ľ��������y(600) = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0)T ��ʾ��ѵ�����ݼ��еĵ�600��ѵ��ʵ��������Ӧ����ȷ����ǣ�����5 ����Ϊ������y(600)�еĵ�5��Ԫ��Ϊ1����������Ԫ��Ϊ0����������Ҫע����ǣ�������y(i) ��10��Ԫ��Ϊ1ʱ ��������0��

##### BP(back propagation)�㷨

BP�㷨������ ����������Ĵ��ۺ������ݶȡ�

�����ݶȣ�����������ƫ�����������ƫ�������ǿ����ô�ͳ����ѧ��������ƫ��������ѧ���㹫ʽ����⡣�������󦨷ǳ���ʱ������Ҫ�����ؽ��м����ˣ���BP�㷨�������������������������⡣BP�㷨�ֳƷ��򴫲��㷨����������㿪ʼ��������㷽���ԡ�ĳ����ʽ���ļ��㣬�õ�һ�顰���ݡ������������ݸպþ�����������Ҫ�� �ݶȡ�

##### Sigmoid �����ĵ���

����ΪʲôҪ����Sigmoid������Sigmoid�����ĵ�����һ���ص㣬��Sigmoid�ĵ���������Sigmoid�����Լ���������ʾ�����£�

$$ g(z) = sigmoid(z) = \frac {1} {1 + e^{-z}} $$

$$ g^`(z) = g(z) (1 - g(z)) $$

##### ѵ��������ʱ�ġ�symmetry������---�����ʼ��������Ĳ�������(Ȩֵ����)

�����ʼ���������󣬾��ǶԲ�������(L)�е�ÿ��Ԫ�أ�����ظ�ֵ��ȡֵ��Χһ��Ϊ[�� ,-��]��

���轫��������(L) �����е�Ԫ�س�ʼ��0���� ���ݼ��㹫ʽ��a1(2) = ��(1) * a(2) �е�ÿ��Ԫ�ض���ȡ��ͬ��ֵ��

��ˣ������ʼ���ĺô����ǣ���ѧϰ ����Ч��

�����ʼ����Matlabʵ�����£����Կ����������ȵ��� randInitializeWeights.m �ж���Ĺ�ʽ���г�ʼ���ġ�Ȼ���ٽ� initial_Theta1 �� initial_Theta2 unroll ��������

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

##### BP�㷨�ľ���ִ�в������£�

����ÿһ��ѵ��ʵ��(x, y)�����á�ǰ�򴫲��㷨������� activations��a(2)�� a(3)����Ȼ���ٶ�ÿһ�����һ���в��j(L) (error term)��ע�⣺�����(input layer)����Ҫ����в

###### �����ÿһ��Ĳв���㹫ʽ���£��������е�������ֻ��3�㣬���ز����ĿΪ1��

* ���������Ĳв���㹫ʽ���£������������ǵ�1�㣬���ز��ǵ�2�㣬������ǵ�3��

$$ \delta_k^{(3)} = a_k^{(3)} - y^{(k)} $$

>�в��(3)��һ���������±� k ��ʾ���������еĵ� k ��Ԫ�ء�yk ����ǰ���ᵽ�ı�ʾ �������������y �еĵ� k ��Ԫ�ء�(��������� y ����ѵ���������Ľ������y�ָ�õ���)

>����ļ�����ʽ��������ʾΪ����(3)= a(3) - y����˦�(3)ά����a(3)һ������������������Ŀ�йء��ڱ����У���10���ʦ�(3)��һ��10*1����

* �������ز�Ĳв���㹫ʽ���£�

$$ \delta_k^{(2)} = (\theta^{(2)})^T\delta_k^{(3)} .* g^`(z^{(2)}) $$

��ÿһ��Ĳв�����֮�󣬾Ϳ��Ը��� ��(delta) �����ˣ���(delta) ������ ������������ͬ��ά������ʼʱ��(delta) �����е�Ԫ��ȫΪ0.

���Ķ���(���㹫ʽ)���£�

$$ \Delta^{(l)} = \Delta^{(l)}+\delta_k^{(l+1)} (a^{(l)})^T $$

�������(L+1)��һ����������(a(1))T��һ������������˺󣬵õ�����һ������
����� ��(delta) ����󣬾Ϳ������������� ���ۺ����ĵ����ˣ���ʽ���£�

$$ \frac {\partial } {\partial \Theta_{ij} ^{(l)}} J(\Theta) = D_{ij} ^{(l)} = \frac 1m \Delta_{ij} ^{(l)}$$

##### �ݶȼ��(gradient checking)

�ݶȼ���ԭ�����£���������ͨ��BP�㷨��������ķ�ʽ����˴��ۺ����ĵ�������������������ȷ�أ�����Ϳ����� �ߵ���ѧ ����ĵ����Ķ���(���޵Ķ���)�����㵼����Ȼ���ٱȽϣ���BP�㷨��õĵ��� �� �õ����Ķ��� ��õĵ��� �����֮��Ĳ�ࡣ

��������(���޶���)---����ʽ���壬���£�

$$ f_i(\theta) \approx \frac {J(\theta^{(i+\epsilon)}) - J(\theta^{(i-\epsilon)})} {2\epsilon}  $$

��������ͨ������ֱ�Ӽ���ķ�ʽ �������ܴ�����ʽѵ��ʱ��Ҫ�ǵùر� gradient checking

���gradient checking ����ж��߼�������Ľ��������ȣ��� BP�㷨�������������ġ�

##### �����������

������������ԣ����ı��������ǿ�����׳��� overfitting problem����һ����Ҫ���򻯡����򻯾��Ǽ���һ��������Ϳ����ˡ�ע�� bias unit����Ҫ����

$$ \frac {\partial } {\partial \Theta_{ij} ^{(l)}} J(\Theta) = D_{ij} ^{(l)} = \frac 1m \Delta_{ij} ^{(l)} \ (for j=0) $$

$$  \frac {\partial } {\partial \Theta_{ij} ^{(l)}} J(\Theta) = D_{ij} ^{(l)} = \frac 1m \Delta_{ij} ^{(l)} + \frac \lambda m \Theta_{ij} ^{(l)}(for j\ge1) $$

##### ʹ��Matlab�� fmincg ���� ���յõ� ��������
```
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
```

##### sigmoidGradient��������
```matlab
# sigmoidGradient.m
g = sigmoid(z) .* (1 - sigmoid(z));
```

##### nnCostFunction��������
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

%��5000���Ľ��label ����y ת����Ԫ��ֻΪ0��1 �ľ���Y
Y = zeros(num_labels, m); %10*5000, each column is a label result
for i = 1:num_labels
    Y(i, y==i)=1;
end

Y1 = 1 - Y;
res1 = 0;
res2 = 0;
for j = 1:m
    %���������ÿһ�����,�ٰѽ����͡�Ԥ��ֵ�ͽ��label��Ӧ��Ԫ�����,����ĳ������x �Ĵ���
    tmp1 = sum( log(a_super3(:,j)) .* Y(:,j) );
    res1 = res1 + tmp1; % m ��֮��
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
    error_2 = ( err_2(2:end) ) .*  sigmoidGradient(z2);% ȥ�� bias unit ��Ӧ�� error units
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

##### ����������������£�
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