# KMP算法需要预先计算一个数组，这个数组的下标是【模式串前缀子串】的结尾字符下标
# 元素内容：最长可匹配【模式串前缀子串】后缀的结尾字符下标
# e.g. 模式串 a b a b a c
# 模式串前缀子串       prefix数组
# a                    prefix[0] = -1
# a b                  prefix[1] = -1
# a b a                prefix[2] = 0
# a b a b              prefix[3] = 1
# a b a b a            prefix[4] = 2
# a b a b a c          prefix[5] = -1

def get_prefix_move(b:str):
    b_len = len(b)
    prefix = [-1 for i in range(len(b)-1)]
    k = -1
    # 如果好前缀只有1位字符，就没有必要考虑模式串前缀后缀重合了，因此从下标1开始计算
    # 模式串最后一个字符也没有必要算进去，因为那样都完全匹配了，没必要用到
    for i in range(1,b_len-1):
        # 不断回溯k, 找到前面找过的最长前缀，然后判断下一位是否等于新增的b[i]
        while(k!=-1 and b[k+1]!=b[i]):
            k = prefix[k]
            # 若b[0, r-1] 是 b[0, i-1]的可匹配最长前缀子串(即，b[0, r-1]是b[0, i-1]的前缀子串也是后缀子串，对所有满足条件的r，取r的最大值)
            # 且b[i]==b[r], 则b[0, r] 也是b[0, i]的可匹配最长前缀子串
        if b[k+1] == b[i]:
            k += 1
        prefix[i] = k
    return prefix

def KMP(a:str, b:str):
    a_len = len(a)
    b_len = len(b)
    j = 0
    # 预先计算prefix数组
    prefix = get_prefix_move(b)
    for i in range(a_len):
        # 当匹配到不相同字符时，若前缀子串存在，则查找prefix数组，找到合适的前缀子串内部重合区域，更新模式串的当前指向指针j
        while(j>0 and a[i]!=b[j]):
            j = prefix[j-1] + 1
        if a[i] == b[j]:
            j += 1
        # 若模式串指向指针已经超出模式串，说明匹配完毕
        if j == b_len:
            return i - b_len + 1
    return -1


            

res_ = KMP('aaabbbbbbaaraaaaaar', 'aar')
print(res_)