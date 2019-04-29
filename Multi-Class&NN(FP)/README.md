# Multi-Class & Neural Networks(feedforward propagation)

ʹ���߼��ع������������(neural networks)ʶ����д�İ���������(0-9)��

## Multi-Class
#####ʹ���߼��ع���ʵ�ֶ��������(one-vs-all)
��ν��������⣬��ָ����Ľ��Ϊ�������ϡ����磬Ԥ��������������Ϊ���ࣺ��(��y==1��ʾ)����(��y==2��ʾ)����(��y==3��ʾ)

�����˼�룬��ʵ���߼��ع����(Ĭ����ָ�����࣬binary classification)�����ƣ��ԡ����족���з���ʱ������������(���������)��Ϊһ�ࣺ(������)���������Ͱ�һ�����������ת�����˶��������⡣ʾ��ͼ���£���ͼ�е�ԲȦ ��ʾ��������ĳһ��� ���������ࣩ

����N��������(N>=3)������ҪN�����躯��(Ԥ��ģ��)��Ҳ����ҪN��ģ�Ͳ����ȣ���һ����һ��������

Ȼ�󣬶���ÿ������ʵ��������ʹ��ÿ��ģ��Ԥ�������ѡȡ���ֵ��������ģ������Ӧ��Ԥ������Ϊ���ս����

��Ϊģ�͵����ֵ����sigmoid���������£���ʵ��һ������ֵ����ע�⣺h��(1)(x)��h��(2)(x)��h��(3)(x)���� ģ�Ͳ����� һ���ǲ�ͬ�ġ����磺
>h��(1)(x)����� Ԥ��Ϊ����(y==1)�ĸ���
h��(2)(x)����� Ԥ��Ϊ����(y==2)�ĸ���
h��(3)(x)����� Ԥ��Ϊ����(y==3)�ĸ���

���������ʶ���������ֵ����⣬һ����Ҫѵ����10���߼��ع�ģ�ͣ�ÿ���߼��ع�ģ�Ͷ�Ӧ��ʶ������һ�����֡�

����һ����5000��������������Ԥ����ֵ���ǣ�y=(1,2,3,4,5,6,7,8,9,10)������ 10 ���� ����0

����ʹ��Matlab fmincg�⺯�� �����ʹ�ô��ۺ���ȡ��Сֵ�� ģ�Ͳ�����


##### matlab�ļ�
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

##### oneVsall��������
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

�������һ�� forѭ����
num_labels Ϊ��������������10����ÿ��������(ģ��)����ʶ��10�������е�ĳһ����

����һ����5000��������ÿ��������400��������������ˣ�ģ�Ͳ����� ������401��Ԫ�ء�

initial_theta = zeros(n + 1, 1); % ģ�Ͳ����ȵĳ�ʼֵ(n == 400)

all_theta��һ��10*401�ľ���ÿһ�д洢��һ��������(ģ��)��ģ�Ͳ����� ������ִ������forѭ�����͵���fmincg�⺯�� ����� ����ģ�͵Ĳ����� �����ˡ�

�����ÿ��ģ�͵Ĳ��������ȣ��Ϳ����� ѵ���õ�ģ����ʶ�������ˡ�����һ����������������(400�� feature variables) input instance��ÿ��ģ�͵ļ��躯��h��(i)(x) ���һ��ֵ(i = 1,2,...10)��ȡ��10��ֵ�����ֵ�Ǹ�ֵ����Ϊ���յ�ʶ����������g(h��(8)(x))==0.96 ���������е� g(h��(i)(x)) (i = 1,2,...10,�� i ������8) ������ʶ��Ľ��Ϊ ���� 8

##### lrcostFunction��������
```matlab
#lrcostFunction.m
m = length(y);
J = 0;
grad = zeros(size(theta));
J = ( log( sigmoid(theta'*X') ) * y + log( 1-sigmoid(theta'*X') ) * (1 - y) )/(-m) + (lambda / (2*m)) * ( ( theta( 2:length(theta) ) )' * theta(2:length(theta)) );
grad = ( X' * ( sigmoid(X*theta)-y ) )/m + ( lambda / m ) * ( [0; ones( length(theta) - 1 , 1 )].*theta );
```

##### predictOneVsAll��������
```matlab
#predictOneVsAll.m
[~,p] = max( X * all_theta',[],2); % �����(X*all_theta')ÿ�е����ֵ��p ��¼����ÿ�е����ֵ������
```

##### �����߼��ع�����������£�
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
�����߼��ع������Է��ࣨ���ļ��躯����һ�����Ժ��������ǻ�һ��ֱ�ߣ������ݷֳ������ࡣ�ɲο���ƪ�����еģ���ʹ���߼��ع���ʵ�ֶ��������(one-vs-all) ���� ��ͼƬ��

����һЩ���ӵ�����߼��ع�ͽ�������ˡ�(�޷�ͨ�� ��ֱ�� �� ��ͬ��� �ֿ���


�������磬���ܹ�ʵ�ֺܸ��ӵķ����Է������⡣

������������ԣ�ͬ����һ��ѵ���������� X��ͬʱ����һ��ģ�Ͳ��� Theta ����ͨ��ĳ���㷨��ģ�Ͳ�������ѵ����֮��(��� Theta ����)����ʹ��ǰ�򴫲��㷨( feedforward propagation algorithm)���о������Ǿ�����ˣ����Ϳ��Զ�����Ĳ�����������Ԥ���ˡ�

����ҵ�У� ģ�Ͳ��� Theta �������Ѿ�ѵ�����˵ģ�ֱ�� load ��Matlab�м��ɡ�

##### predict��������
```matlab
#predict.m
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);% p �� 5000*1����

% ģ��ʵ��ǰ�򴫲��㷨
X = [ones(m, 1) X];
a_super_2 = sigmoid(Theta1 * X');
a_super_2 = [ones(1,m); a_super_2];% add bias unit
a_super_3 = sigmoid(Theta2 * a_super_2);

[~,p] = max( a_super_3' ,[], 2 ); % �������Ľ������Ԥ�⣬���߼��ع��Ԥ�����ƣ�ѡȡ��������ֵ ��Ϊ���յ�Ԥ����
```
��������ͨ��Matlab �� max ��������þ��� a_super3�� ��ÿһ�е����ֵ����ÿһ�е��е����ֵ ������ ��ֵ������p�����У�a_super3�� ��һ��5000�г�10�еľ��� 
����p����Ԥ��Ľ�������������� a_super3�� ��10�У��� p ��ÿ��Ԫ�ص�ȡֵ��ΧΪ[1,10]�����ֱ���������� 0-9

##### �������������򴫲��������£�
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