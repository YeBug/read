# 机器学习：  
  
## 监督学习  
  1.回归问题：描述连续的问题  
  2.分类问题：离散的问题  
## 无监督学习  
  聚类问题  
  
### 目标：针对特定实际的问题建立强泛化能力的模型  
## 避免过拟合与欠拟合问题  
有监督学习中，预测误差的来源主要有两部分，分别为 bias  与 variance，模型的性能取决于 bias 与 variance 的 平衡  
理解 bias 与 variance 有助于我们诊断模型的错误，避免 over-fitting 或者 under-fitting.  
Bias：度量了学习算法的期望输出与真实结果的偏离程度, 刻画了算法的拟合能力，Bias 偏高表示预测函数与真实结果差异很大。  
Variance：则代表“同样大小的不同的训练数据集训练出的模型”与“这些模型的期望输出值”之间的差异。训练集变化导致性能变化， Variance 偏高表示模型很不稳定。  
Noise：刻画了当前任务任何算法所能达到的期望泛化误差的下界，即刻画了问题本身的难度。  
regularization:使得模型曲线更平滑，但是随着权重的增加可能会伤害bias，导致测试集erro更大  
Error=Bias^2+Variance+Noise  
underfitting:erro 受bias影响大，训练集与模型差距大,需要redesign model  
overfitting:erro 受variance影响大，测试集与模型差距大，需要increase data/regularization  
  
## 梯度下降：  
为了寻找建模函数中参数的最佳值，实现最佳拟合所设计的一种算法  
从一个起始点w计算受其影响的loss函数的梯度，沿着梯度的反向对w进行移动，判断此时loss函数的梯度变化，如此迭代多次找到loss函数梯度变化趋近于0的点。  
### 下面是一个简单的线性拟合案例1:
```
import numpy as np
import matplotlib.pyplot as plt

x_train = [100,80,120,75,60,43,140,132,63,55,74,44,88]
y_train = [120,92,143,87,60,50,167,147,80,60,90,57,99]
b=0
w=0
lr=0.00001
m=len(x_train)
diff=[0,0]
def model(x):
    return  x*w+b
error0=0
error1=0
#退出迭代的两次误差差值的阈值
epsilon=0.000001

while True:
    diff = [0, 0]
    for i in range(m):
        diff[0]+=model(x_train[i])-y_train[i]
        diff[1]+=(model(x_train[i])-y_train[i])*x_train[i]
    b=b-lr*diff[0]
    w=w-lr*diff[1]
    error1 = 0
    for i in range(len(x_train)):
        error1 += (y_train[i] - (b + w * x_train[i])) ** 2
    print(error1)
    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1

plt.plot(x_train,y_train,'bo')
plt.plot(x_train,[model(x) for x in x_train])
plt.show()
```
![result](https://github.com/YeBug/read/blob/master/1537026574.jpg)   
    
### 简单线性回归，二维自变量，案例2：   
```
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[1, 2],[2, 1],[2, 3],[3, 5],[1, 3],[4, 2],[7, 3],[4, 5],[11, 3],[8, 7]])
y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26])
a=0
b=0
c=0
lr=0.001
m=len(x_train)
diff=[0,0,0]

def model(x):
    return  x[0]*a+x[1]*b+c

error0=0
error1=0
#退出迭代的两次误差差值的阈值
epsilon=0.01
while True:
    diff = [0,0,0]
    for i in range(m):
        diff[0]+=(model(x_train[i])-y_train[i])*x_train[i][0]
        diff[1]+=(model(x_train[i])-y_train[i])*x_train[i][1]
        diff[2]+=model(x_train[i])-y_train[i]
    a=a-lr*diff[0]
    b=b-lr*diff[1]
    c=c-lr*diff[2]
    error1 = 0
    for i in range(len(x_train)):
        error1 += (y_train[i] - model(x_train[i])) ** 2
    print(error1)
    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1
    plt.plot([model(x) for x in x_train])

#plt.plot(x_train,y_train,'bo')
plt.plot([model(x) for x in x_train])
plt.show()
```
下面是learning rate 较小的时候的拟合结果  
![small](https://github.com/YeBug/read/blob/master/1537027380.jpg)  
下面是调整learning rate 后的拟合结果  
![big](https://github.com/YeBug/read/blob/master/1537027362.jpg)  
   
梯度下降技巧：  
1.从较大的learning rate进行建模  
2.adagrad算法调整learning rate；  
3.采用随机样本进行随机梯度下降  
4.feature scaling，让不同特征值具有相同的缩放程度  

问题：找到的可能是鞍点或局部最小点  

评估模型泛化能力：划分训练集与测试集  
划分方法：1.留出法  2.交叉验证法 3.自助法

## 分类问题  
朴素贝叶斯分类：  
通过贝叶斯定理，以后验概率推算先验概率模型，根据训练数据进行概率计算  
例：二元分类下的朴素贝叶斯分类  
二分类C1,C2;分类对象x有{a1,a2}两个特征属性；已知P(C1)+P(C2)=1；根据数据求得训练集中的P(C1),P(C2)；  
目标概率为某一对象x’属于C1的概率，即P(C1|X')；根据贝叶斯定理需要求得P(X'|C1)；  
当特征属性是离散分布时，P(X'|C1)=P(a1'|C1)\*P(a2'|C1)  
当特征属性是连续分布时，P(X'|C1)即为a1,a2在高斯分布模型g(ai,u,m)下的积,其中，最佳参数u,m的计算有公式  
得到必要概率后代入贝叶斯公式即可求得P(C1|X')=P(X'|C1)\*P(c1)/P(X)  
p28
