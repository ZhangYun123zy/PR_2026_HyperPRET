import scipy.io as sio
from  sklearn import preprocessing
import  numpy as np

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)