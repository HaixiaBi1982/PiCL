import copy
import numpy as np

def dataset_standardization(dataset, mean, std):
    new_dataset = copy.deepcopy(dataset)
    for data in range(len(dataset.data)):
        for i in range(6):
            new_dataset.data[data][i, :, :] = (dataset.data[data][i, :, :] - mean[i]) / std[i]
    return new_dataset


def get_dataset_mean_and_std(dataset):
    data_array = np.array(copy.deepcopy(dataset.data))
    mean = np.zeros([6], dtype=np.complex64)
    std = np.zeros([6], dtype=np.complex64)

    for i in range(6):
        mean[i] = np.mean(data_array[:,i,:,:])

        if type(data_array[:,i,:,:]) == complex:
            std[i] = np.sqrt(((data_array[:,i,:,:] - mean) * np.conj(data_array[:,i,:,:] - mean)).sum() / np.size(data_array[:,i,:,:]))
        else:
            std[i] = np.std(data_array[:,i,:,:])
    return mean, std
