import math
import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt, colors
from numpy import float32, sqrt
import h5py
from torch.utils.data import Dataset
import copy
from colormap import rgb2hex


def SlidingWindow(image, stride, window_size, window_center=None, padding=True, channel=False, print_show=True):
    # Input image (C, W, H), input label (W, H)
    # zero padding
    if channel:
        name = "data"
        a = numpy.zeros((6, window_size, window_size), dtype=numpy.complex64)

    else:
        name = "labels"
        a = numpy.zeros((window_size, window_size), dtype=int)
    width = image.shape[-2]
    height = image.shape[-1]
    total_num = math.ceil(width / stride) * math.ceil(height / stride)
    # total_num: SlidingWindow Indicates the total number of samples intercepted

    if padding:
        if window_center is None:
            if window_size % 2:
                W_padding_ahead = W_padding_behind = H_padding_ahead = H_padding_behind = int(window_size / 2)
                window_center = (W_padding_ahead + 1, H_padding_ahead + 1)

            else:
                W_padding_ahead = H_padding_ahead = int(window_size / 2)
                W_padding_behind = H_padding_behind = int(window_size / 2) - 1
                window_center = (W_padding_ahead, H_padding_ahead)
        else:
            H_padding_ahead = window_center[1]
            H_padding_behind = window_size - window_center[1] - 1
            W_padding_ahead = window_center[0]
            W_padding_behind = window_size - window_center[0] - 1

        if channel:
            image = numpy.pad(image, ((0, 0), (W_padding_ahead, W_padding_behind), (H_padding_ahead, H_padding_behind)),
                              'constant', constant_values=0)
        else:
            image = numpy.pad(image, ((W_padding_ahead, W_padding_behind), (H_padding_ahead, H_padding_behind)),
                              'constant',
                              constant_values=0)
    position = []
    b = []
    num = 0
    for i in range(0, width, stride):
        for j in range(0, height, stride):
            for m in range(window_size):
                for n in range(window_size):
                    if channel:
                        a[:, m, n] = image[:, i + m, j + n]
                    else:
                        a[m, n] = image[i + m, j + n]
            position.append((i, j))
            b.append(copy.deepcopy(a))
            num += 1
            if print_show:
                print("\rCollecting {} ...{:.2%} total:{}".format(name, num / total_num, num), end="")
    print(" Done! total:{}".format(num))
    return b, window_center, position


def get_mean_and_std(T):
    mean = np.zeros([6], dtype=np.complex64)
    std = np.zeros([6], dtype=np.complex64)
    for i in range(T.shape[0]):
        mean[i] = np.mean(T[i, :, :])

        if type(T[i, :, :]) == complex:
            std[i] = sqrt(((T[i, :, :] - mean) * np.conj(T[i, :, :] - mean)).sum() / np.size(T[i, :, :]))
        else:
            std[i] = np.std(T[i, :, :])

    return mean, std


class DataSet(Dataset):
    """   PolSAR dataset  """
    """ class_num: The total number of classes in the dataset """
    """ dataset_path: The folder path where the dataset is located """
    """ delete_0: Remove data with label 0 when initializing the dataset """
    """ window_size: The size of the sliding window """
    """ stride: The step size when sliding the window across the entire image to extract sample data """
    """ window_center (input tuple): The center of the sample to be extracted (default is window_size/2) """
    """ std: Whether the dataset is standardized or not """

    def __init__(self, dataset_path=None, delete_0=True, window_size=12, stride=1, window_center=(6, 6), std=True,
                 print_show=True, class_num=0, return_position=False):
        self.delete_0 = delete_0
        self.return_position = return_position
        self.window_size = window_size
        self.std = std

        if dataset_path is not None:
            if 'Flevo1991' in dataset_path:
                self.name = 'Flevo1991'
                class_num = 14
            elif 'FlveoBig' in dataset_path:
                self.name = 'FlveoBig'
                class_num = 15
            elif 'Oberpfa' in dataset_path:
                self.name = 'Oberpfa'
                class_num = 3
            elif 'San900690' in dataset_path:
                self.name = 'San900690'
                class_num = 5
            else:
                self.name = None
                assert 0
            self.class_num = class_num

            total_data, total_labels = ReadData(dataset_path, std=std)
            self.total_data = total_data
            self.total_labels = total_labels

            self.shape = total_labels.shape

            data_list, _, _ = SlidingWindow(total_data, stride, window_size, window_center, channel=True,
                                            print_show=print_show)
            labels_list, window_center, data_position = SlidingWindow(total_labels, stride, window_size, window_center,
                                                                      print_show=print_show)

            data_list_without_0 = []
            labels_list_without_0 = []
            data_position_without_0 = []

            if delete_0:
                for i in range(len(labels_list)):
                    if labels_list[i][window_center] != 0:
                        data_list_without_0.append(data_list[i])
                        labels_list_without_0.append(labels_list[i])
                        data_position_without_0.append(data_position[i])

                print("Delete class 0！ Remained:{}".format(len(data_list_without_0)))
                self.data = data_list_without_0
                self.labels = labels_list_without_0
                self.position = data_position_without_0

            else:
                print("You haven't delete samples of 0 class!")
                self.data = data_list
                self.labels = labels_list
                self.position = data_position

        else:
            self.data = []
            self.labels = []
            self.position = []
            self.class_num = class_num
            self.shape = (0, 0)
            self.total_data = np.zeros([6, self.shape[0], self.shape[1]])
            self.total_labels = np.zeros(self.shape)
        self.window_center = window_center

    def __getitem__(self, index):
        z = torch.torch.as_tensor(self.data[index])  # nparray --> torch
        label = copy.deepcopy(self.labels[index][self.window_center])

        label -= 1  # classes 1-15 should be mapping to classes 0-14
        if (not 0 <= label <= self.class_num) and self.delete_0:
            print("label={}:should 0 <= label <= self.class_num !!!".format(label))
            assert 0

        one_hot_label = one_hot(label, num_classes=self.class_num)

        if self.return_position:
            p = self.position[index]
            return z, one_hot_label, p
        else:
            return z, one_hot_label

    def __len__(self):
        return len(self.data)

    def __get_mean_and_std__(self):
        # The mean and variance of all data in this dataset are obtained
        mean, std = get_mean_and_std(self.total_data)
        return mean, std

    def __get_position__(self):
        # Get the position coordinates of all samples on image
        return self.position

    def __get_class_num__(self):
        return self.class_num

    def __get_size__(self):
        return self.shape

    def __get_whole_image__(self):
        # Returns the data and labels of the entire graph
        return self.total_data, self.total_labels

    def __get_name__(self):
        return self.name

    def __get_window_center__(self):
        return self.window_center


def one_hot(label, num_classes):
    ones = torch.eye(num_classes) * (1 + 1j)
    onehot = ones[label, :]
    return onehot.numpy()


def standardization(data):
    mean = numpy.mean(data)
    if type(data) == complex:
        std = sqrt(((data - mean) * numpy.conj(data - mean)).sum() / numpy.size(data))
    else:
        std = numpy.std(data)
    return (data - mean) / std


def load_matrix_from_file(file_path, load_list, shape, dtype, San900690=False):
    data_list = []
    for i in range(len(load_list)):
        load_path = file_path + load_list[i]
        data = ReadFile(load_path, shape, San900690)
        data_list.append(data)
    T11 = data_list[0]
    T12 = data_list[1] + 1j * data_list[2]
    T13 = data_list[3] + 1j * data_list[4]
    T22 = data_list[5]
    T23 = data_list[6] + 1j * data_list[7]
    T33 = data_list[8]
    return T11.astype(dtype), T12.astype(dtype), T13.astype(dtype), T22.astype(dtype), T23.astype(dtype), T33.astype(
        dtype)


def ReadFile(filepath, shape, San900690):
    """ Read each element of the T-matrix in the data set """
    data = numpy.fromfile(filepath, dtype=float32)
    if San900690:
        data = data.reshape((shape[0] * 2, shape[1] * 2))
        data = data[0:data.shape[0] - 1:2, 0:data.shape[1] - 1:2]
    else:
        data = data.reshape(shape)
    return data


def ReadFileFlveoBig(filepath):
    data = np.fromfile(filepath, dtype=float32)
    data = data.reshape(750, 1024)
    return data


def ReadData(dataset_path, std=True, dtype=numpy.complex64):
    """  Read PolSAR data  """
    """  std: Whether to perform Z-Score standardization  """
    """  dtype: output data type (plural np.plex64)  """

    global T11, T12, T13, T22, T23, T33, GT

    if 'FlveoBig' in dataset_path:
        # FlveoBig：750*1024

        GT = h5py.File(dataset_path + "/label.mat", 'r')
        GT = GT['label'][:]
        # 取出主键为data的所有的键值

        # load_list = ["T11.bin", 'T12_imag.bin', "T12_real.bin", "T13_imag.bin", "T13_real.bin", "T22.bin",
        #              "T23_imag.bin", "T23_real.bin", "T33.bin"]

        bin_path = dataset_path + '/LEE_T3/T11.bin'
        T11 = ReadFileFlveoBig(bin_path)
        T11 = T11.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T12_imag.bin'
        T12_imag = ReadFileFlveoBig(bin_path)
        bin_path = dataset_path + '/LEE_T3/T12_real.bin'
        T12_real = ReadFileFlveoBig(bin_path)
        T12 = T12_real + 1j * T12_imag
        T12 = T12.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T13_imag.bin'
        T13_imag = ReadFileFlveoBig(bin_path)
        bin_path = dataset_path + '/LEE_T3/T13_real.bin'
        T13_real = ReadFileFlveoBig(bin_path)
        T13 = T13_real + 1j * T13_imag
        T13 = T13.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T22.bin'
        T22 = ReadFileFlveoBig(bin_path)
        T22 = T22.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T23_imag.bin'
        T23_imag = ReadFileFlveoBig(bin_path)
        bin_path = dataset_path + '/LEE_T3/T23_real.bin'
        T23_real = ReadFileFlveoBig(bin_path)
        T23 = T23_real + 1j * T23_imag
        T23 = T23.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T33.bin'
        T33 = ReadFileFlveoBig(bin_path)
        T33 = T33.transpose(1, 0)

    if std:
        # 是否对T矩阵的输入参数进行标准化
        T11 = standardization(T11)
        T12 = standardization(T12)
        T13 = standardization(T13)
        T22 = standardization(T22)
        T23 = standardization(T23)
        T33 = standardization(T33)
    T_matrix = numpy.dstack((T11, T12, T13, T22, T23, T33))  # (1024, 750, 6)
    T_matrix = numpy.flip(T_matrix, 1)
    GT = numpy.flip(GT, 1)
    T_matrix = numpy.transpose(T_matrix, (2, 0, 1))  # 重新指定轴的顺序:(6, W, H)

    return T_matrix, GT

def set_colors_bar(dataset_name):
    """Set color bar"""
    global colors_list, label_list, sm

    if 'FlveoBig' in dataset_name:
        colors_list = ['white', rgb2hex(0, 0, 225), 'darkred', 'indigo', 'red', 'm', 'forestgreen', 'darkgoldenrod', 'lime',
                               'orange', 'cyan', rgb2hex(200,188,252), 'pink', 'navajowhite', 'yellow', 'lightgreen']

        label_list = ['', '1:Water', '2:Barley', '3:Peas', '4:Stembeans', '5:Beet', '6:Forest',
                      '7:Bare soil', '8:Grass',
                      '9:Rapeseed', '10:Lucerne', '11:Wheat', '12:Wheat 2', '13:Buildings', '14:Potatoes', '15:Wheat 3']

    assert len(colors_list) == len(label_list)
    # It is used to display Chinese labels properly
    plt.rcParams['font.sans-serif'] = ['SimHei']
    c_map = colors.LinearSegmentedColormap.from_list('mylist', colors_list, N=len(colors_list))
    sm = plt.cm.ScalarMappable(cmap=c_map)
    return sm, label_list, c_map


def draw_ground_truth(full_dataset):
    _, GT = full_dataset.__get_whole_image__()
    dataset_name = full_dataset.__get_name__()

    x = numpy.arange(GT.shape[0])
    y = numpy.arange(GT.shape[1])
    x, y = numpy.meshgrid(x, y)
    z = GT[x, y]
    sm, label_list, c_map = set_colors_bar(dataset_name)

    plt.figure(figsize=(GT.shape[0] / 100, GT.shape[1] / 100))
    plt.pcolor(x, y, z, cmap=c_map)
    plt.title("{}\nImage size：{}\n".format(dataset_name, GT.shape), fontsize=20)

    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.grid(True)

    cbar = plt.colorbar(sm, ticks=numpy.linspace(0, 1, len(label_list), endpoint=False) + 1 / (2 * len(label_list)),
                        label='Classes', )
    cbar.ax.set_yticklabels(label_list)
    cbar.ax.axes.tick_params(length=0)

    plt.show()