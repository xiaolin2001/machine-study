# 导入numpy模块
import numpy as np


# 定义sigmoid函数
def sigmoid(x):
    """
    输入：
    x：数组
    输出：
    z：经过sigmoid函数计算后的数组
    """
    z = 1 / (1 + np.exp(-x))
    return z


# 定义参数初始化函数
def initialize_params(dims):
    """
    输入：
    dims：参数维度
    输出：
    z：初始化后的参数向量W和参数值b
    """
    # 将权重向量初始化为零向量
    W = np.zeros((dims, 1))
    # 将偏置初始化为零
    b = 0
    return W, b


# 定义对数几率回归模型主体
def logistic(X, y, W, b):
    """
    输入：
    X: 输入特征矩阵
    y: 输出标签向量
    W: 权重系数
    b: 偏置参数
    输出：
    a: 对数几率回归模型输出
    cost: 损失
    dW: 权重梯度
    db: 偏置梯度
    """
    # 训练样本量
    num_train = X.shape[0]
    # 训练特征数
    num_feature = X.shape[1]
    # 对数几率回归模型输出
    a = sigmoid(np.dot(X, W) + b)
    # 交叉熵损失
    cost = -1 / num_train * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    # 权重梯度
    dW = np.dot(X.T, (a - y)) / num_train
    # 偏置梯度
    db = np.sum(a - y) / num_train
    # 压缩损失数组维度
    cost = np.squeeze(cost)
    return a, cost, dW, db


# 定义对数几率回归模型训练过程
def logistic_train(X, y, learning_rate, epochs):
    """
    输入：
    X: 输入特征矩阵
    y: 输出标签向量
    learning_rate: 学习率
    epochs: 训练轮数
    输出：
    cost_list: 损失列表
    params: 模型参数
    grads: 参数梯度
    """
    # 初始化模型参数
    W, b = initialize_params(X.shape[1])
    # 初始化损失列表
    cost_list = []
    # 迭代训练
    for i in range(epochs):
        # 计算当前迭代的模型输出、损失和参数梯度
        a, cost, dW, db = logistic(X, y, W, b)
        # 参数更新
        W = W - learning_rate * dW
        b = b - learning_rate * db
        # 记录损失
        if i % 100 == 0:
            cost_list.append(cost)
        # 打印训练过程中的损失
        if i % 100 == 0:
            print('epoch %d cost %f' % (i, cost))
    # 保存参数
    params = {
        'W': W,
        'b': b
    }
    # 保存梯度
    grads = {
        'dW': dW,
        'db': db
    }
    return cost_list, params, grads


# 定义预测函数
def predict(X, params):
    """
    输入：
    X: 输入特征矩阵
    params: 训练好的模型参数
    输出：
    y_pred: 转换后的模型预测值
    """
    # 模型预测值
    y_pred = sigmoid(np.dot(X, params['W']) + params['b'])
    # 基于分类阈值对概率预测值进行类别转换
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入生成分类数据函数
from sklearn.datasets._samples_generator import make_classification

# 生成100×2的模拟二分类数据集
X, labels = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=2)
# 设置随机数种子
rng = np.random.RandomState(2)
# 对生成的特征数据添加一组均匀分布噪声
X += 2 * rng.uniform(size=X.shape)
# 标签类别数
unique_labels = set(labels)
# 根据标签类别数设置颜色
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# 绘制模拟数据的散点图
for k, col in zip(unique_labels, colors):
    x_k = X[labels == k]
    plt.plot(x_k[:, 0], x_k[:, 1], 'o',
             markerfacecolor=col,
             markeredgecolor='k',
             markersize=14)
plt.title('Simulated binary data set')
plt.show();

# 按9∶1简单划分训练集与测试集
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], labels[:offset]
X_test, y_test = X[offset:], labels[offset:]
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
print('X_train =', X_train.shape)
print('X_test =', X_test.shape)
print('y_train =', y_train.shape)
print('y_test =', y_test.shape)

# 执行对数几率回归模型训练
cost_list, params, grads = logistic_train(X_train, y_train, 0.01, 1000)
# 打印训练好的模型参数
print(params)
# 基于训练参数对测试集进行预测
y_pred = predict(X_test, params)
print(y_pred)

# 导入classification_report模块
from sklearn.metrics import classification_report

# 打印测试集分类预测评估报告
print(classification_report(y_test, y_pred))


### 绘制对数几率回归分类决策边界
def plot_decision_boundary(X_train, y_train, params):
    '''
    输入：
    X_train: 训练集输入
    y_train: 训练集标签
    params：训练好的模型参数
    输出：
    分类决策边界图
    '''
    # 训练样本量
    n = X_train.shape[0]
    # 初始化类别坐标点列表
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    # 获取两类坐标点并存入列表
    for i in range(n):
        if y_train[i] == 1:
            xcord1.append(X_train[i][0])
            ycord1.append(X_train[i][1])
        else:
            xcord2.append(X_train[i][0])
            ycord2.append(X_train[i][1])
    # 创建绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制两类散点，以不同颜色表示
    ax.scatter(xcord1, ycord1, s=32, c='red')
    ax.scatter(xcord2, ycord2, s=32, c='green')
    # 取值范围
    x = np.arange(-1.5, 3, 0.1)
    # 分类决策边界公式
    y = (-params['b'] - params['W'][0] * x) / params['W'][1]
    # 绘图
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


plot_decision_boundary(X_train, y_train, params)

# 导入对数几率回归模块
from sklearn.linear_model import LogisticRegression
# 拟合训练集
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# 预测测试集
y_pred = clf.predict(X_test)
# 打印预测结果
print(y_pred)