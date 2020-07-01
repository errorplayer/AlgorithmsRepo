#!/usr/bin/env python
# coding: utf-8


# 生成坏字符规则所需的哈希表
# 需要特别注意模式串和主串中出现的所有元素，必须都包含在hash表中
# 这个hash表的设计不一定按照下面函数generateBadCharacter(),只要符合要求即可
# 需求：创建一个坏字符hash表，下标是字符的ascii码值，元素内容为该ascii码对应的字符最靠近模式串尾部的下标，若不存在则存 -1
def generateBadCharacter(b:str, table_size=256) -> list:
    bc = [-1 for i in range(table_size)]
    for idx in range(len(b)):
        ascii_value = ord(b[idx])
        if ascii_value > table_size:
            raise ValueError("The ascii value of the character exceeds the table size! ")
        bc[ascii_value] = idx
    return bc

# 生成好后缀规则的prefix和suffix数组
# suffix idx: 字符串长度，所以下标为0的元素为-1 undefined
#        ele: 模式串去除最后一个字符后，与[以下标为长度的模式串后缀子串]完全重合的部分的首字符下标
# prefix idx: 字符串长度，所以下标为0的元素为-1 undefined
#        ele: 以下标为长度的模式串前缀恰好也是模式串后缀，则该元素为True，否则为False
def generateGoodPrefixSuffix(b:str):
    b_len = len(b)
    suffix = [-1 for i in range(b_len)]
    prefix = [False for i in range(b_len)]
    for i in range(b_len-1):
        j = i
        k = 0
        while(j>=0 and b[j]==b[b_len-1-k]):
            j -= 1
            k += 1
            suffix[k] = j + 1
        # j 为当前比较中坏字符的下标，
        # 若为-1则说明从模式串的第0位开始符合条件，因此模式串前缀匹配模式串后缀，prefix数组的当前长度下标的元素应置True
        if j == -1: prefix[k] = True
    return prefix, suffix

# 根据好后缀规则应该移动的位数计算
def moveByGoodSuffixRule(j:int, b_len:int, prefix:[], suffix:[]):
    # j 为坏字符在模式串中的下标
    # k 为好后缀长度
    k = b_len-1-j
    # 如果模式串前面有和好后缀重合的字符串，则直接移动至使其重合
    if suffix[k] != -1: return j - suffix[k] + 1
    # 若没有，则需要往后大幅度移动模式串， 这时候选择模式串前缀字符串中于好后缀子串重合的部分
    # 肯定是越长越好，所以从长的开始找
    for r in range(1, k):
        if prefix[b_len-r] == True: return r
    return b_len

# 正式进入BM算法
def bm(a:str, b:str):
    a_len = len(a)
    b_len = len(b)
    bc = generateBadCharacter(b)
    prefix, suffix = generateGoodPrefixSuffix(b)
    i = 0
    # 如果模式串的末尾已经超过主串，则没必要检查是否匹配了，肯定不匹配
    while(i<=a_len-b_len):
        j = b_len-1
        # 坏字符规则
        while(j >= 0 and a[j+i] == b[j]):
            j -= 1
        # 检查到坏字符的下标为j , 若j 小于0， 则说明已经匹配成功，因为坏字符已经没有出现在模式串中了
        if j < 0: return i
        # 根据坏字符规则，计算出需要后移的step,但是注意，光使用坏字符规则可能产生step为负的情况
        bc_step = j - bc[ord(a[i+j])]
        gs_step = 0
        # 如果存在好后缀，则进入好后缀规则的执行
        if j < b_len-1:
            gs_step = moveByGoodSuffixRule(j, b_len, prefix, suffix)
        # 由于好后缀和坏字符两个规则相互独立，在最终确定后移step时需要选择最大的，应为坏字符给出的step可能为负数
        i += max(bc_step, gs_step)
    return -1
            

res_ = bm('adfgdbfckkabfrt', 'kabfrt')
print(res_)


