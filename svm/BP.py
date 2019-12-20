import numpy as np
from sklearn.preprocessing import LabelBinarizer
import svm

def sigmoid(x):   #激活函数
    return 1/(1+np.exp(-x))

def dsigmoid(x):  #激活函数的导数
    return x*(1-x)

class NeuralNetwork:
    def __init__(self,layers):
        # 权重初始化,最后一列为偏置值
        self.V = np.random.random((layers[0] + 1, layers[1]+1))*2 - 1
        self.W = np.random.random((layers[1] + 1, layers[2])) * 2 - 1

    def train(self,X,y,lr=0.11,epochs=10000):  #lr为学习率  epochs为训练轮数
        for n in range(epochs+1):
            # 在训练集中随机选取一行
            i = np.random.randint(X.shape[0])
            x = [X[i]]
            # 转为二维数据：由一维一行转为二维一行
            x = np.atleast_2d(x)

            # L1：输入层传递给隐藏层的值；输入层400个节点，隐藏层600个节点
            # L2：隐藏层传递到输出层的值；输出层10个节点
            L1 = sigmoid(np.dot(x, self.V))
            L2 = sigmoid(np.dot(L1, self.W))

            # L2_delta：输出层对隐藏层的误差改变量
            # L1_delta：隐藏层对输入层的误差改变量
            L2_delta = (y[i] - L2) * dsigmoid(L2)
            L1_delta = L2_delta.dot(self.W.T) * dsigmoid(L1)

            # 计算改变后的新权重
            self.W += lr * L1.T.dot(L2_delta)
            self.V += lr * x.T.dot(L1_delta)

            #每训练1000次输出一次准确率
            if n%1000 == 0:
                    print('迭代次数：', n, )
                    self.test()

    def test(self):
        predictions = []
        y_test = [int(x) for x in Y]
        for j in range(X.shape[0]):
            x = np.atleast_2d(X[j])

            L1 = sigmoid(np.dot(x, self.V))
            L2 = sigmoid(np.dot(L1, self.W))

            # 将最大的数值所对应的标签返回
            predictions.append(L2.argmax())

        # np.equal()：相同返回true，不同返回false
        accuracy = np.mean(np.equal(predictions, y_test))
        print('准确率：', accuracy)

if __name__ == '__main__':
    # 定义神经网络节点数目
    NN = NeuralNetwork([400, 600, 10])   #注意mnist原始图片大小为28*28，这里为了和原来实验相同，改成了20*20

    # 读取图像数据
    X, Y = svm.read_all_data()

    # 输入数据归一化：当数据集数值过大，乘以较小的权重后还是很大的数，代入sigmoid激活函数就趋近于1，不利于学习
    X -= X.min()
    X /= X.max()

    # 添加偏置值：最后一列全是1
    temp = np.ones([X.shape[0], X.shape[1] + 1])
    temp[:, 0:-1] = X
    X = temp

    # 标签二值化：将十进制标签转为二进制
    labels_train = LabelBinarizer().fit_transform(Y)

    print('开始训练')
    NN.train(X, labels_train, epochs=10000)
    print('训练结束')
