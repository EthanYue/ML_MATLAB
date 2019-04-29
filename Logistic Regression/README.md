# Logistic Regression

���߼��ع����ѧ���Ŀ��Գɼ����жϸ�ѧ���Ƿ������ѧ��


**�ܵ�˼·Ϊ���Ƶ��������������ݶ��½���������������ϵ�Theta�����ͨ��Theta������Ϻ��ֱ�ߣ�ͬʱ���ü���ɱ�����������ݶ��½������еĳɱ�ֵ��**

###### sigmoid function
���� logistic regression���ԣ�����Ե��� classification problem������ֻ���۶��������⣬��������ġ����ݳɼ���ѧ�������ֻ�����֣�y==0ʱ���ɼ�δ�ϸ񣬲�����ѧ��y==1ʱ������ѧ������y�����Ҫô��0��Ҫô��1
������� linear regression�����ļ��躯���������ģ�

$$ h_\theta(x) = \theta^Tx $$

���躯����ȡֵ������ԶԶ����1��Ҳ����ԶԶС��0�����������ܵ�һЩ����������Ӱ�졣��������ͼ�У���ֻ��Լ���������躯�����ڵ���0.5ʱ��Ԥ��y==1��С��0.5ʱ��Ԥ��y==0��

�����������sigmoid function���Ϳ��԰Ѽ��躯����ֵ��Լ������[0, 1]֮�䡣��֮������sigmoid function�����ܹ����õ���Ϸ��������е����ݣ���������Ƕȿ���regression model �� linear model ������ classification problem.
����sigmoid�󣬼��躯������

$$ g(z) = \frac {1} {1 + e^{-z}}  $$

�������ռ��躯��Ϊ��

$$ h_\theta(x) = \frac {1} {1 + e^{-\theta^Tx}} $$

###### ���ۺ���(cost function)
ѧϰ���̾���ȷ�����躯���Ĺ��̣�����˵�ǣ���� �� �Ĺ��̡�
�����ȼ��� �� �Ѿ�������ˣ�����Ҫ�ж���õ�������躯�����׺ò��ã�����ʵ��ֵ��ƫ���Ƕ��٣���ˣ����ô��ۺ���������

$$ J(\theta) = -\frac 1 m [log(g(\theta^Tx^T))y + log(1 - g(\theta^Tx^T))(1 - y)] $$

###### �ݶ��½��㷨(Gradient descent algorithm)
�ݶ��½��㷨��������������(ƫ����)������˵�ǣ���������������������ķ���--�ݶȷ����½�����졣
������֪��������ĳЩͼ��������ĺ������������кܶ������Ϊ0�ĵ㣬���ຯ����Ϊ��͹����(non-convex function)����ĳЩ��������ֻ��һ��ȫ��Ψһ�ĵ���Ϊ0�ĵ㣬��Ϊ convex function
convex function�ܹ��ܺõ���Gradient descentѰ��ȫ����Сֵ������ͼ��ߵ�non-convex�Ͳ�̫����Gradient descent�ˡ�
������Ϊ�������ԭ��logistic regression �� cost function����д�������������ʽ

$$ \frac{\partial  J(\theta)}{\partial \theta} = \frac 1m [x^T g(x\theta ) - y)] $$

###### �߼��ع�����򻯣�Regularized logistic regression��
���򻯾���Ϊ�˽�����������(overfitting problem)��
һ����ԣ���ģ�͵�����(feature variables)�ǳ��࣬��ѵ����������Ŀ(training set)�ֱȽ��ٵ�ʱ��ѵ���õ��ļ��躯��(hypothesis function)�ܹ��ǳ��õ�ƥ��training set�е����ݣ���ʱ�Ĵ��ۺ�������Ϊ0����ͼ�����ұߵ��Ǹ�ģ�� ����һ������ϵ�ģ�͡�
������Ϊ feature variable�ǳ��࣬���� hypothesis function ���ݴκܸߣ�hypothesis function��úܸ��ӣ�������ϵ����߷���Ϊ��

$$ y = \theta_0 + \theta_1x + \theta_2 x^2 + \theta_3 x^3 + \theta_4 x^4 $$

������һ��"������"������ ���ݴε�����������Ӱ�죬�� hypothesis function�ͻ���ƽ��������ǰ���ᵽ���ݶ��½��㷨��Ŀ������С��cost function�������ڰ� theta(3) �� theta(4)��ϵ������Ϊ1000����úܴ���ƫ����ʱ����Ӧ�صõ���theta(3) �� theta(4) �Ͷ�Լ����0�ˡ�
��һ��أ����Ƕ�ÿһ��theta(j)��j>=1���������򻯣��͵õ���һ�����µĴ��ۺ��������е� lambda(��)�ͳ�Ϊ���򻯲���(regularization parameter)���������򻯺�ĳɱ�����Ϊ��

$$ J(\theta) = -\frac {1} {2m} [log(g(\theta^Tx^T))y + log(1 - g(\theta^Tx^T))(1 - y) + \lambda \sum_{i=1}^n \theta_j ^2] $$

�������J(theta)���Կ��������lambda(��)=0�����ʾû��ʹ�����򻯣����lambda(��)����ʹ��ģ�͵ĸ�����������ú�С������h(x)=theta(0)���Ӷ����Ƿ��ϣ����lambda(��)��С����δ��������򻯵�Ч������ˣ�lambda(��)��ֵҪ���ʡ����򻯺���ݶȹ�ʽ���£�

$$ \frac{\partial  J(\theta)}{\partial \theta} = \frac 1m [x^T g(x\theta ) - y)]\ (for j = 1)$$

$$ \frac{\partial  J(\theta)}{\partial \theta} = \frac 1m [x^T g(x\theta ) - y)] + \frac \lambda m \theta_j \ (for j \ge 1)$$

>�����ѵ������(training instance)��ѧ�������ο��Գɼ����Լ�TA�Ƿ��ܹ���ѧ�ľ�����y=0��ʾ�ɼ����ϸ񣬲���¼ȡ��y=1��ʾ¼ȡ����ˣ���Ҫ����trainging set ѵ����һ��classification model��Ȼ���������classification model ��������ѧ���ܷ���ѧ��

>ѵ�����ݵĳɼ��������£���һ�б�ʾ��һ�ο��Գɼ����ڶ��б�ʾ�ڶ��ο��Գɼ��������б�ʾ��ѧ�����0--������ѧ��1--������ѧ��

>��ʷ�������£���һ�б�ʾ�����˿�������λΪ���ˣ��ڶ��б�ʾ���󣬵�λΪ10,000$
```
34.62365962451697, 78.0246928153624,  0
30.28671076822607, 43.89499752400101, 0
35.84740876993872, 72.90219802708364, 0
60.18259938620976, 86.30855209546826, 1
....
....
....
```

ͨ������costfunction.m�ļ��ж����coustFunction�������Ӷ������ݶ��½��㷨�ҵ�ʹ���ۺ���J(theta)��С���� �߼��ع�ģ�Ͳ���theta������costFunction�����Ĵ������£�
```
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```
�������������һ�п��Կ�����������ͨ�� fminunc ���� costFunction����������� theta�ģ��������Լ�ʹ�� Gradient descent ��for ѭ���������� theta��
ͨ��Gradient descent�㷨�����theta����theta���뵽���躯���У��͵õ��� logistic regression model��
###### matlab�ļ�
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

###### plotData��������
```matlab
# plotData.m
pos = find(y==1);
neg = find(y==0);
plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
```

###### sigmoid��������
```matlab
# sigmoid.m
g = 1./(ones(size(z)) + exp(-z)); % ������� ��ʾ 1 ���Ծ���(����)�е�ÿһ��Ԫ��
```

###### costFunction��������
```matlab
#costFunction.m
m = length(y);
J = 0;
grad = zeros(size(theta));
J = (log(sigmoid(theta'*X')) * y + log(1-sigmoid(theta'*X')) * (1 - y))/(-m);
grad = (X' * (sigmoid(X*theta)-y))/m;
```

###### costFunctionReg��������
```matlab
#costFunctionReg.m
m = length(y);
J = 0;
grad = zeros(size(theta));
J = ( log( sigmoid(theta'*X') ) * y + log( 1-sigmoid(theta'*X') ) * (1 - y) )/(-m) + (lambda / (2*m)) * ( ( theta( 2:length(theta) ) )' * theta(2:length(theta)) );
grad = ( X' * ( sigmoid(X*theta)-y ) )/m + ( lambda / m ) * ( [0; ones( length(theta) - 1 , 1 )].*theta );
```

###### ����costFunctionReg.m�Ĵ������£�
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