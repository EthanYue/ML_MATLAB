# Linear Resression

>����ϰ�У���Ҫʵ��һ�������������Իع顣������һ����ʷ����<�����˿ڣ���������>������ҪԤ�����ĸ������п�������ȽϺã�

>**�ܵ�˼·Ϊ���Ƶ��������������ݶ��½���������������ϵ�Theta�����ͨ��Theta������Ϻ��ֱ�ߣ�ͬʱ���ü���ɱ�����������ݶ��½������еĳɱ�ֵ��**

>###### ���躯��(hypothesis function)
�ڸ���һЩ��������(training set)�󣬲���ĳ��ѧϰ�㷨(learning algorithm)���������ݽ���ѵ�����õ���һ��ģ�ͻ���˵�Ǽ��躯��������ҪԤ�������ݵĽ��ʱ������������Ϊ���躯�������룬���躯�������õ����������������ΪԤ��ֵ��

>$$ h(z) = \sum_{n=0}^n \theta_ix_i = \theta^Tx $$

>###### ���ۺ���(cost function)
ѧϰ���̾���ȷ�����躯���Ĺ��̣�����˵�ǣ���� �� �Ĺ��̡�
�����ȼ��� �� �Ѿ�������ˣ�����Ҫ�ж���õ�������躯�����׺ò��ã�����ʵ��ֵ��ƫ���Ƕ��٣���ˣ����ô��ۺ���������

> $$ J(\theta) = \frac 1 m \sum_{n=0}^n h_\theta(x^{(i)} - y^{(i)})^2 $$

>###### �ݶ��½��㷨(Gradient descent algorithm)
�ݶ��½��㷨�ı��ʾ�����ƫ��������ƫ��������0����� ��

> $$ \theta_j := \theta_j - \alpha \frac{\partial }{\partial \theta} J(\theta) = \theta_j - \frac \alpha m x^T(h(x^{(i)}) - y^{(i)}) = \theta_j - \frac \alpha m x^T(\theta^T x - y) $$

>###### ���ڷ�������ͻع�����
���躯����������y��predicted y�������ֱ�ʾ��ʽ����ɢ��ֵ��������ֵ�����籾���н�����Ԥ��������������������������ֵ���ٱ���˵������ʷ���������Ԥ����������������� or �����꣩����Ԥ��Ľ��������ɢ��ֵ(discrete values)����ˣ���hypothesis function�����������ֵ���������ѧϰ����Ϊ�ع�����(regression problem)�����������ɢ��ֵ�����Ϊ��������(classification problem)

>��ʷ�������£���һ�б�ʾ�����˿�������λΪ���ˣ��ڶ��б�ʾ���󣬵�λΪ10,000$


>```
5.5277    9.1302
8.5186   13.6620
7.0032   11.8540
.....
......
```

###### matlab�ļ�

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


###### plotData��������
```matlab
# plotData.m
plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');

```

###### gradientDescent��������
```matlab
#gradientDescent.m
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
	theta = theta - (alpha/m)*X'*(theta'*X-y);
	J_history(iter) = computeCost(X, y, theta);
end
```


###### computeCost��������
```matlab
#computeCost.m
predictions = X * theta;
sqrErrors = (predictions-y) .^ 2;
J = 1/(2*m) * sum(sqrErrors);
```

###### �������ֱ��
```
theta = gradientDescent(X, y, theta, alpha, iterations);
hold on; 
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
```