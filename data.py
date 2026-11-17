
import modelStatsRecord
import numpy as np

# 打开 .txt 文件
with open('ddd.txt', 'r') as file:
    # 逐行读取文件内容并将每行分割成元素
    lines = file.readlines()
    # 初始化一个空列表来存储数据
    data = []
    for line in lines:
        # 将每行的元素以空格分割，并转换为浮点数
        row = [float(x) for x in line.strip().split()]
        # 将每行数据添加到数据列表中
        data.append(row)

# 将数据列表转换为 numpy 矩阵
A = np.array(data)

# 打印矩阵
print("矩阵 A:")
print(A)

classes_num=9
ITER=5
ELEMENT_ACC_RES_SS4=A[:classes_num,:]/100
AA_RES_SS4=A[classes_num,:]/100
OA_RES_SS4=A[classes_num+1,:]/100
KAPPA_RES_SS4=A[classes_num+2,:]/100

ITER=5
modelStatsRecord.outputRecord(ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
                                ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
                             classes_num, ITER,
                              'Utopia_5shot_my.txt')