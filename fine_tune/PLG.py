import copy
import torch
import numpy as np
import pandas as pd
import seaborn
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import matplotlib.pyplot as plt

def SLIC(img_path, n_segments,
         compactness=100,
         show_img=False):
    np.set_printoptions(threshold=np.inf)
    img = io.imread(img_path)
    if show_img:
        plt.imshow(img)
        plt.show()
    segments = slic(img, n_segments=n_segments, compactness=compactness)
    out = mark_boundaries(img, segments)
    if show_img:
        print("n_segments={}, compactness={}".format(n_segments, compactness))
        plt.axis('off')
        plt.imshow(out)
        plt.show()
    return segments


def getSuperpixelsData(segment_label, classes, segments, test_data, class_num):
    l = []
    for x in range(segments.shape[1]):
        for y in range(segments.shape[0]):
            if segments[y, x] == segment_label:
                l.append((x, y))

    data = SlidingWindow(l, test_data)
    labels = []
    assert 1 <= classes <= class_num
    for i in range(len(data)):
        labels.append(torch.tensor(np.ones([12, 12], dtype=int) * classes))

    return data, labels, l


def SlidingWindow(superpixels_list, image, window_size=12, padding=True):
    a = np.zeros((6, window_size, window_size), dtype=np.complex64)

    if padding:
        if window_size % 2:
            padding_ahead = padding_behind = int(window_size / 2)
        else:
            padding_ahead = int(window_size / 2)
            padding_behind = int(window_size / 2) - 1

        image = np.pad(image, ((0, 0), (padding_ahead, padding_behind), (padding_ahead, padding_behind)),
                       'constant', constant_values=0)

    data_list = []
    window_size = int(window_size / 2)
    for p in superpixels_list:
        for x in range(-window_size, window_size):
            for y in range(-window_size, window_size):
                a[:, window_size + x, window_size + y] = image[:, x + p[0], y + p[1]]

        data_list.append(torch.tensor(a))
    return data_list


# This function determines a list of a few classes
def count_classnum_and_def_minorityclasses(dataset):
    global label_list
    class_num = dataset.__get_class_num__()
    sample_center = dataset.__get_window_center__()

    n = [0] * class_num
    for i in range(len(dataset)):
        label = dataset.labels[i][sample_center] - 1
        assert 0 <= label <= class_num - 1
        n[int(label)] += 1

    minority_classes_l = []
    c = 0
    for p in np.array(n) / len(dataset):
        if p < 0.5 / class_num:
            minority_classes_l.append(c)
        c = c + 1
    print("Minority class is {}".format(minority_classes_l))
    return minority_classes_l


def aug_dataset_by_SLIC(dataset,
                        show_img,
                        thresholds_of_PLG_in_each_class,
                        classes_list=[],
                        sample_center=(6, 6),
                        n_segments_size=5):
    print("%" * 40)
    print(f"thresholds_of_PLG_in_each_class = {thresholds_of_PLG_in_each_class}")
    print("data_aug_with_SLIC:")

    aug_dataset = copy.deepcopy(dataset)
    aug_dataset.data = []
    aug_dataset.labels = []
    aug_dataset.position = []
    position = dataset.__get_position__()

    if len(classes_list) == 0:
        print("Minority classes_list in empty !!!")
        print("%"*30)
        return aug_dataset
    assert len(classes_list) != 0

    class_num = dataset.__get_class_num__()
    dataset_name = dataset.__get_name__()
    class_position = []
    for i in range(class_num):
        class_position.append([])

    for i in range(len(dataset.data)):
        classes = dataset.labels[i][sample_center] - 1
        assert 0 <= classes <= 14
        if classes in classes_list:
            class_position[classes].append(position[i])

    if 'Flevo1991' == dataset_name:
        PauliRGB_img_path = "../benchmark_dataset/Flevo1991/Flevoland_PauliRGB.png"
    else:
        assert 0
    shape = dataset.__get_size__()
    n_segments = int((shape[0] * shape[1]) / (n_segments_size * n_segments_size))
    segment = SLIC(img_path=PauliRGB_img_path, n_segments=n_segments, show_img=show_img)
    dict_of_segment_label_and_p = {}
    for c in range(len(class_position)):
        for p in class_position[c]:
            segment_label = segment[p[1], p[0]]
            dict_of_segment_label_and_p.setdefault(segment_label, [])
            dict_of_segment_label_and_p[segment_label].append(c)

    for key in list(dict_of_segment_label_and_p.keys()):
        if len(set(dict_of_segment_label_and_p[key])) > 1:
            print("No.{} segment super_pixel (with classes{}) is deleted !!!".format(key,
                                                                                     dict_of_segment_label_and_p[key]))
            del dict_of_segment_label_and_p[key]
            continue

    count = np.zeros([class_num])
    total_data, _ = dataset.__get_whole_image__()

    for key, value in dict_of_segment_label_and_p.items():
        assert len(set(value)) == 1
        c = list(set(value))[0]
        while True:
            if count[c] < thresholds_of_PLG_in_each_class[c]:
                datas, labels, position = getSuperpixelsData(key, c + 1, segment, total_data, class_num)
                aug_dataset.data.extend(datas)
                aug_dataset.labels.extend(labels)
                aug_dataset.position.extend(position)
                count[c] += len(datas)
            else:
                break

    for i in range(class_num):
        print("class{} aug: {}".format(i, int(count[i])))
    print("Total aug: {}".format(int(count.sum())))
    print("%" * 40)
    return aug_dataset
