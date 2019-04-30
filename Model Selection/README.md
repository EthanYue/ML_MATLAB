# Debugging a learning algorithm

����ˮ������ˮ����(water level) ʹ�����򻯵����Իع�ģ��Ԥ ˮ����(water flowing out of dam)��Ȼ�� debug ѧϰ�㷨 �Լ� ����ƫ��ͷ���� �����Իع�ģ�͵�Ӱ�졣

>����ҵ�����ݼ��ֳ������֣�
* ѵ����(training set)����������(ѵ����)��X�������ǩ(label of result)���� y
* ������֤��(cross validation set)��ȷ�����򻯲��� Xval �� yval
* ���Լ�(test set) for evaluating performance�����Լ��е����� �Ǵ�δ������ ѵ�����е�

>ѵ������һ����12��ѵ��ʵ����ÿ��ѵ��ʵ��ֻ��һ��������

##### ���Ի���ѧϰ�㷨�ķ���
* Ѱ�Ҹ����ѵ������
* ����ѵ��������������
* ����ѵ�������������
* ���Ӷ���ʽ�Ĵ���
* �����
* ��С��

##### ѵ��������֤�������Լ�
����ģ�͵Ĺ����У�Ϊ����֤��ͬѵ��ģ�͵�׼ȷ�ԣ�Ӧ�ö�ѵ�����ģ�ͽ��в��ԣ������ֱ���ڲ��Լ��ϲ��Բ�ͬ��ģ�ͣ���������ʧ��С��ģ�ͽ�ֻ�����ĳһ�ض��Ĳ��Լ���������Ҫ��ѵ�����ݷֳ�ѵ��������֤�������Լ������֣�����Ϊ60%��20%��20%����ѵ�����ģ������֤���Ͻ�����֤���Դ���ѡ����ģ�ͽ����߷����ԡ�

##### ��ϸ�ƫ�Bias���͸߷��Variance��
��ƫ�ΪǷ��ϣ�Underfitting�����߷��Ϊ����ϣ�Overfitting������ƫ��͸߷������Ҫ���֣�
* ����ʽ��������߽ײ㣨d��

$$ h(\theta) = \theta_0 + \theta_1x + \theta_2x^2 + \theta_nx^d $$

$$ J(\theta) = \frac 1 {2m} \sum_{i=1}^n (h(x^{(i)}) - y^{(i)}) ^2$$
* 
 * d������߹�С�����ᵼ��ѵ��������֤������ʧ���󣻵�d��С��ѵ��������֤������ʧ��Խ�ӽ���d����ѵ��������֤������ʧ��Խ��

* ���򻯵Ĳ��� ��

$$ Reg(\theta) = \frac {\lambda} {2m} \sum_{j=1}^m \theta_j^2$$

* 
 * �� ����ᵼ���ݶ��½���theta��С���Ӷ�����ʽ��d��С�����ָ�ƫ��� ����ᵼ���ݶ��½���theta���󣬴Ӷ�����ʽ��d��󣬳��ָ߷��
 * �� ������߹�С�����ᵼ��ѵ��������֤������ʧ���󣻵��˹�С��ѵ��������֤������ʧ��Խ�󣬦˹���ѵ��������֤������ʧ��Խ�ӽ���

* ѧϰ���ߣ�ѵ���������� m
 * ��ƫ������£����� m ����ѵ��������֤������ʧ���С������С��һ���̶Ⱥ�ѵ��������֤������ʧ��������ƽ�������ᷢ���仯�����Դ�ʱ����ѵ������������Ч��
  * �߷�������£����� m ����ѵ��������֤������ʧ���С������С��һ���̶Ⱥ� ѵ��������֤������ʧ�������һ�νϴ�Ĳ�࣬������ѵ�������������󣬸ò���Խ��ԽС

##### ���Է����ܽ�
* �����ڸ�ƫ�����
 * ����ѵ����������
 * ������϶���ʽ�Ľײ�
 * ��С ��
* �����ڸ߷������
 * ��Сѵ����������
 * ��С��϶���ʽ�Ľײ�
 * ���� ��


##### linearRegCostFunction��������
```matlab
# linearRegCostFunction.m
reg = (lambda/(2*m)) * ((theta(2:length(theta)))' * theta(2:length(theta)));
J = sum((X * theta- y).^ 2) / (2*m) + reg;


grad_tmp = X' * (X*theta - y) ./ m;
grad = [grad_tmp(1:1); grad_tmp(2: end) + (lambda/m) * theta(2:end)];
```

##### learningCurve��������
```matlab
# learningCurve.m
for i = 1:m
        theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
        error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
        error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end
```

##### polyFeatures��������
```matlab
# polyFeatures.m
for i = 1:p
    X_poly(:,i) = X.^i;
end
```

##### validationCurve��������
```matlab
# validationCurve.m
for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    theta = trainLinearReg(X, y, lambda);
    error_train(i) = linearRegCostFunction(X, y, theta, 0);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end
```