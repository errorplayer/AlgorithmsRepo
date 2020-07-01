import numpy as np

def train(filename):

    # 中文分词有四种状态：Begin，Middle，End，Single (character)
    # Begin：开始的字
    # Middle：中间的字
    # End: 结束字
    # Single: 单个字
    # 举例：我|喜欢|打|乒乓球
    #               S     B   E   S      B M E
    #               3     0   2   3     0 1  2
    status2index = {'B':0, 'M':1, 'E':2, 'S':3}

    # 状态转移概率矩阵
    A = np.zeros((4, 4))

    # 观测概率矩阵
    # 下面将会用到ord函数，这是python自带的，能把字符转换成编码
    # 中文编码不超过65536
    B = np.zeros((4, 65536))

    # 初始概率矩阵
    PI = np.zeros(4)

    with open(filename, encoding='utf-8') as file:
        for line in file.readlines():
            word_status = []
            words = line.strip().split()

            for i, word in enumerate(words):
                start_character = ''
                if (len(word) == 1):
                    start_character = 'S'
                    word_status.append('S')
                    B[status2index['S']][ord(word)] += 1
                else:
                    start_character = 'B'
                    if len(word) == 2:
                        word_status.append('B')
                        word_status.append('E')
                        B[status2index['B']][ord(word[0])] += 1
                        B[status2index['E']][ord(word[1])] += 1
                    else:
                        word_status.append('B')
                        B[status2index['B']][ord(word[0])] += 1
                        for idx in range(1, len(word)-1):
                            word_status.append('M')
                            B[status2index['M']][ord(word[idx])] += 1
                        word_status.append('E')
                        B[status2index['E']][ord(word[-1])] += 1
                if i == 0:
                    PI[status2index[start_character]] += 1

            for i in range(1, len(word_status)):
                A[status2index[word_status[i-1]]][status2index[word_status[i]]] += 1


    sum_pi = sum(PI)
    for i in range(len(PI)):
        if PI[i] == 0:
            PI[i] = -3.14e+100
        else:
            PI[i] = np.log(PI[i] / sum_pi)

    for i in range(len(A)):
        tmp_sum = sum(A[i])
        for j in range(len(A[i])):
            if A[i][j] == 0:
                A[i][j] = 999
            else:
                A[i][j] = np.log(A[i][j] / tmp_sum)

    for i in range(len(A)):
        tmp_min = min(A[i]) * 0.1
        for j in range(len(A[i])):
            if A[i][j] == 999:
                A[i][j] = tmp_min



    for i in range(len(B)):
        tmp_sum = sum(B[i])
        for j in range(len(B[i])):
            if B[i][j] == 0:
                B[i][j] = 999
            else:
                B[i][j] = np.log(B[i][j] / tmp_sum)
    for i in range(len(B)):
        tmp_min = min(B[i]) * 0.1
        for j in range(len(B[i])):
            if B[i][j] == 999:
                B[i][j] = tmp_min

    return (PI, A, B)

def partition(hmm_param, article):
    PI, A, B = hmm_param
    result = []

    for line in article:
        delta = [[0 for i in range(4)] for _ in range(len(line))]
        psi = [[0 for i in range(4)] for _ in range(len(line))]

        for t in range(len(line)):
            if t == 0:
                psi[t] = [0, 0, 0, 0]
                for i in range(4):
                    delta[t][i] = PI[i] + B[i][ord(line[t])]

            else:
                for i in range(4):
                    tmp = [delta[t-1][j]+A[j][i] for j in range(4)]

                    delta[t][i] = max(tmp) + B[i][ord(line[t])]
                    psi[t][i] = tmp.index(max(tmp))

        status_best_path = []

        status_tmp = delta[-1].index(max(delta[-1]))
        status_best_path.append(status_tmp)

        for t in range(len(line)-2, -1, -1):
            status_tmp = psi[t+1][status_best_path[0]]
            status_best_path.insert(0, status_tmp)

        line_partition = ''

        for t in range(len(line)):
            line_partition += line[t]
            if status_best_path[t] == 2 or status_best_path[t] == 3:
                if t != len(line)-1:
                    line_partition += '|'
        result.append(line_partition)
    return result


def file2article(filename):
    article = []
    with open(filename, encoding='utf-8') as file:
        for line in file.readlines():
            tmp_line = line.strip()
            article.append(tmp_line)
    return article



if __name__ == '__main__':

    param = train('HMMTrainSet.txt')

    # artl = file2article('test.txt')

    # partition_result = partition(param, artl)
    # for _ in range(len(partition_result)):
    #     print(partition_result[_])

    print('**********自定义测试***************')
    line_num=int(input('请输出测试语句行数 = '))

    for i in range(line_num):
        sentence=input('请输入语句：')
        article_cumstmize_partition=partition(param,[sentence])
        print(article_cumstmize_partition)











