import numpy as np
import math
import time
import pandas as pd

np.random.seed(100)
def safe_equal(a, b):
    if abs(a-b) < 0.000001:
        return True
    return False

def loadData(fileName):
    data=pd.read_csv(fileName,header=None)
    #将dataframe转化为numpy.array
    data=data.values

    #从样本中切分出分类结果
    y_label = data[:, 0]
    x_label = data[:, 1:]/255 #转化为0-1之间的数
    np.random.shuffle(x_label)
    np.random.shuffle(y_label)

    # 使用SVM进行二分类
    # 将数据分为两类，大于0为1，等于0为-1
    # y_label[y_label>0]=1
    # y_label[y_label==0]=-1

    # 按照5分界，样本点很难找出一个较好的超平面区分开，正确率只有80多
    y_label[y_label<5]=-1
    y_label[y_label>=5]=1

    return x_label,y_label

class SVM:
    def __init__(self, X, Y, threshold=1e-3, sigma=10, C=10.0):
        self.x = X
        self.y = Y
        self.sample_size, self.feature_dim = X.shape
        self.t = threshold
        self.sigma = sigma
        self.C = C
        self.alpha = [0 for _ in range(self.sample_size)]
        self.b = 0
        self.E = [-Y[i] for i in range(self.sample_size)]
        self.K = self.calc_Kernel()
        self.supportVector = []

    def GaussianKernel(self, xi, xj):
        return np.exp(-1.0 * (np.linalg.norm(xi - xj)) ** 2 / (2 * (self.sigma ** 2)))

    def calc_Kernel(self):
        K = [[0 for _ in range(self.sample_size)] for __ in range(self.sample_size)]
        progress = 1
        for i in range(self.sample_size):

            for j in range(i, self.sample_size):
                xi = self.x[i]
                xj = self.x[j]
                Kij = self.GaussianKernel(xi, xj)
                K[i][j], K[j][i] = Kij, Kij

            if i % (self.sample_size * (progress / 10)) == 0:
                print(f'Kernel construction {progress}0%')
                progress += 1

        print('Kernel construction 100%')

        return K

    def calc_Gx(self, index_i):
        sum = 0
        for idx in range(self.sample_size):
            if self.alpha[idx] > 0:
                sum += self.alpha[idx] * self.y[idx] * self.K[idx][index_i]
        return sum + self.b

    def calc_E(self, index_i):
        # 根据 eq7.105
        Gx_i = self.calc_Gx(index_i)
        Yi = self.y[index_i]
        self.E[index_i] = Gx_i-Yi
        return Gx_i-Yi

    def Is_break_KKT(self, index_i):
        Gx_i = self.calc_Gx(index_i)
        Yi = self.y[index_i]
        alpha_i = self.alpha[index_i]
        Gyi = Gx_i * Yi

        if abs(alpha_i - 0) <= self.t and Gyi >= 1:
            return False
        if (0 - self.t) < alpha_i < (self.C + self.t) and abs(Gyi - 1) == self.t:
            return False
        if abs(alpha_i - self.C) <= self.t and Gyi <= 1:
            return False

        return True

    def find_alpha2(self, index_i):
        if self.E[index_i] < 0:
            max_value = float('-inf')
            max_j = 0
            for j in range(self.sample_size):
                if index_i != j and self.E[j] > max_value:
                    max_j = j
                    max_value = self.E[j]
            return max_j, self.calc_E(max_j)
        else:
            min_value = float('inf')
            min_j = 0
            for j in range(self.sample_size):
                if index_i != j and self.E[j] < min_value:
                    min_j = j
                    min_value = self.E[j]
            return min_j, self.calc_E(min_j)

    def train(self, epoch=20):
        steps = 0
        alpha_change = 1

        while(steps < epoch and alpha_change > 0):
            alpha_change = 0
            steps += 1
            print(f'epoch = {steps}')
            for i in range(self.sample_size):
                if self.Is_break_KKT(i):
                    E1 = self.calc_E(i)
                    j, E2 = self.find_alpha2(i)

                    Y2 = self.y[j]
                    Y1 = self.y[i]
                    eta = self.K[i][i] + self.K[j][j] + self.K[i][j] * 2
                    alpha1_old = self.alpha[i]
                    alpha2_old = self.alpha[j]

                    alpha2_unc = alpha2_old + Y2 * (E1 - E2) / eta

                    L = 0
                    H = 0
                    if Y1 == Y2:
                        L = max(0, alpha1_old + alpha2_old - self.C)
                        H = min(self.C, alpha1_old + alpha2_old)
                    else:
                        L = max(0, alpha2_old - alpha1_old)
                        H = min(self.C, self.C + alpha2_old - alpha1_old)
                    if safe_equal(L, H):
                        continue

                    if alpha2_unc > H:
                        alpha2_unc = H
                    elif alpha2_unc < L:
                        alpha2_unc = L
                    alpha2_new = alpha2_unc
                    alpha1_new = alpha1_old + Y1 * Y2 * (alpha2_old - alpha2_new)
                    self.alpha[i] = alpha1_new
                    self.alpha[j] = alpha2_new

                    # 根据eq7.115 eq7.116更新b值
                    b1_new = -1.0 * E1 - Y1 * self.K[i][i] * (alpha1_new - alpha1_old) - Y2 * self.K[j][i] * (alpha2_new - alpha2_old) + self.b
                    b2_new = -1.0 * E2 - Y1 * self.K[i][j] * (alpha1_new - alpha1_old) - Y2 * self.K[j][j] * (alpha2_new - alpha2_old) + self.b
                    if 0 < alpha1_new < self.C:
                        self.b=b1_new
                    elif 0 < alpha2_new < self.C:
                        self.b=b2_new
                    else:
                        self.b=(b1_new+b2_new)/2

                    # 只更新本次优化过的两个alpha的Ex
                    self.calc_E(i)
                    self.calc_E(j)

                    if abs(alpha2_old - alpha2_new) >= (1e-3):
                        alpha_change += 1
                        # print(f'changed alpha {alpha_change}')

        for i in range(self.sample_size):
            if self.alpha[i] > 0:
                self.supportVector.append(i)

    def predict(self, x):
        Gx = 0
        for i in self.supportVector:
            Gx += self.y[i] * self.alpha[i] * self.GaussianKernel(x, self.x[i])
        return np.sign(Gx + self.b)

    def test(self,X_test, y_test):
        acc = 0
        for i in range(len(X_test)):
            x = np.array(X_test[i])
            y_pred = self.predict(x)
            if safe_equal(y_pred, y_test[i]):
                acc += 1
        print('now_acc=', acc / len(y_test))
from sklearn import svm



if __name__=="__main__":
    # 获取当前时间
    start = time.time()

    # 读取训练文件
    print('load TrainData')
    X_train, y_train = loadData('./mnist/mnist_train.csv')

    # 读取测试文件
    print('load TestData')
    X_test, y_test = loadData('./mnist/mnist_test.csv')

    print('Init SVM classifier')
    svm=SVM(X_train[0:6000],y_train[0:6000])

    print('start to train')
    svm.train()

    print('start to test')
    svm.test(X_test[0:400], y_test[0:400])
    # clf = svm.SVC()
    # clf.fit(X_train[0:6000],y_train[0:6000])
    # print(clf.score(X_test[0:400], y_test[0:400]))
    # 获取结束时间
    end = time.time()

    print('run time:', end - start)































