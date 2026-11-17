import numpy as np
import time
import collections
from sklearn import metrics
def outputRecord(ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
                 CATEGORY, ITER, path1):
    print_matrix = np.zeros((CATEGORY + 5, ITER + 1), dtype=object)
    print_matrix[0:CATEGORY, 0:ITER] = np.around(ELEMENT_ACC_RES_SS4, 4)
    print_matrix[CATEGORY, 0:ITER] = np.around(AA_RES_SS4, 4)
    print_matrix[CATEGORY + 1, 0:ITER] = np.around(OA_RES_SS4, 4)
    print_matrix[CATEGORY + 2, 0:ITER] = np.around(KAPPA_RES_SS4, 4)
    #print_matrix[CATEGORY + 3, 0:ITER] = np.around(TRAINING_TIME_RES_SS4, 4)
    #print_matrix[CATEGORY + 4, 0:ITER] = np.around(TESTING_TIME_RES_SS4, 4)
    element_mean = np.mean(print_matrix[:, :-1], axis=1)
    element_std = np.std(np.float64(print_matrix[:, :-1]), axis=1)
    for i in range(CATEGORY  + 3):
        print_matrix[i, ITER] = "{:.2f}".format(element_mean[i] * 100) + " ± " + "{:.2f}".format(element_std[i] * 100)
    #for i in range((CATEGORY + 3), (CATEGORY + 5)):
    #    print_matrix[i, ITER] = "{:.2f}".format(element_mean[i]) + " ± " + "{:.2f}".format(element_std[i])
    np.savetxt(path1, print_matrix.astype(str), fmt='%s', delimiter="\t", newline='\n')