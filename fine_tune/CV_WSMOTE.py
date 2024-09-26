import copy
import heapq
import math
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
from dataset_preparing.dataset_def import DataSet


def turn_6channel_2_matrix(channel):
    # 将6通道数据重构为3*3的矩阵
    matrix = np.zeros([3, 3], np.complex64)
    matrix[0, 0] = channel[0]
    matrix[0, 1] = channel[1]
    matrix[0, 2] = channel[2]
    matrix[1, 0] = np.conj(channel[1])
    matrix[1, 1] = channel[3]
    matrix[1, 2] = channel[4]
    matrix[2, 0] = np.conj(channel[2])
    matrix[2, 1] = np.conj(channel[4])
    matrix[2, 2] = channel[5]
    return matrix


def SMOTE(original_dataset, network, net_name, a, save_path, K=5, window_size=12, print_training=True):
    """
    original_dataset: The dataset for oversampling
    network: The network to which the training classification head belongs
    {class_num}: minority class samples; The number of classes of samples that require oversampling
    net_name: The type of network encoder for training,
      0 for real network,
      1 for complex network,
      2 for complex attention network
    a: Number of training epochs conducted
    {K}: Number of nearest; default 5; K nearest
    """
    print("*" * 30)
    print("SMOTE：")

    class_num = original_dataset.__get_class_num__()
    sample_center = original_dataset.__get_window_center__()
    dataset_name = original_dataset.__get_name__()

    Amount_of_SMOTE = []
    list = []
    for i in range(class_num):
        list.append([])

    # Put the original data into a list by class
    for i in range(len(original_dataset)):
        label = original_dataset.labels[i][sample_center] - 1
        assert label >= 0
        list[label].append(torch.as_tensor(original_dataset.data[i]))
    # Count how much data there is in each category
    num = np.zeros(class_num)
    for i in range(len(original_dataset)):
        label = original_dataset.labels[i][6, 6] - 1
        assert label >= 0
        num[label] += 1

    list_of_minority_classes = []
    dict_of_SMOTE_aug = {}
    for classes, aug in dict_of_SMOTE_aug.items():
        list_of_minority_classes.append(classes)
        Amount_of_SMOTE.append(aug)

    reciprocal_num = 1 / num
    n = (reciprocal_num / reciprocal_num.sum()) * num.sum()
   # n: The final number of increments per class
    # (the ratio of the reciprocal to the sum of the reciprocal of all classes)

    for classes in range(class_num):
        list_of_minority_classes.append(classes)
        Amount_of_SMOTE.append(math.ceil(n[classes] / num[classes]))
        # Calculate by n how many times SMOTE you need at least for this class

    smote_index_dict = {}
    aug_amount = 0
    for c in range(len(list_of_minority_classes)):
        classes = list_of_minority_classes[c]
        k = K
        N = Amount_of_SMOTE[c]
        count = 0
        if len(list[classes]) <= k:
            print("Minority classes{}'s amount={} should be more than K={}!!!".format(classes, len(list[classes]), k))
            if len(list[classes]) <= 1:
                continue
            k = len(list[classes]) - 1

        distance = np.zeros((len(list[classes]), len(list[classes])))
        smote_index_list = []
        for i in range(distance.shape[0]):
            # Calculate the wishart distance between samples in the same class
            for j in range(distance.shape[1]):
                distance[i, j] = get_distance(list[classes][i], list[classes][j], network)
                # Calculate Euclidean distance of feature space

            exclude = np.array([i])
            y = copy.deepcopy(distance[i, :])
            y = np.delete(y, exclude)
            # Distance calculations on the diagonal should be excluded

            ##################################################################################
            min_number = heapq.nsmallest(k, y)
            min_index = []
            for t in min_number:
                index = distance[i, :].tolist().index(t)
                #########################################
                if index == i:
                    continue
                assert index != i
                #########################################
                min_index.append(index)

            if int(N) > len(min_index):
                min_index = random.sample(min_index, int(N) % len(min_index)) + int(int(N) / len(min_index)) * min_index
                # If the oversampling multiple int (N) is greater than K,
                # the sample nearest K is oversampled repeatedly
            else:
                min_index = random.sample(min_index, int(N))
                # N of the K nearest samples are randomly selected

            smote_index_list.append(min_index)
            # The i samples should be combined with similar samples in min_index
            count += len(min_index)

        # print("Minority classes{}'s aug: {}".format(classes, count))
        aug_amount += count
        smote_index_dict[classes] = smote_index_list
    return training_network_with_synthetic_data(list, network, smote_index_dict, original_dataset, net_name, a,
                                                class_num, n, window_size, dataset_name, save_path, print_training)


def get_distance(data1, data2, network):
    data1 = torch.as_tensor(data1).unsqueeze(0)
    data2 = torch.as_tensor(data2).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation1 = network(data1.to(device), output_representation=True)
    representation2 = network(data2.to(device), output_representation=True)
    distance = torch.norm(representation1 - representation2)
    return distance


def training_network_with_synthetic_data(data_list, network, smote_index_dict, original_dataset, net_name, a, class_num,
                                         n, window_size, dataset_name, save_path, print_training):
    #################################################
    new_dataset = DataSet(window_size=(6, 6), class_num=class_num)
    # 初始化新的dataset ###############################

    smote_sample_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for c in smote_index_dict.keys():
        smote_sample_dict[c] = []
    for k in smote_index_dict.keys():
        smote_list = []
        amount = 0

        for i in range(len(smote_index_dict[k])):

            representation1 = network(data_list[k][i].unsqueeze(0).to(device), output_representation=True)
            for j in range(len(smote_index_dict[k][i])):
                if amount >= round(n[k]):
                    continue

                representation2 = network(data_list[k][smote_index_dict[k][i][j]].unsqueeze(0).to(device),
                                          output_representation=True)
                gap = random.random()
                dif = representation2 - representation1
                smote = representation1 + gap * dif

                amount += 1
                smote_list.append(smote.squeeze(0))
        print("Minority classes{}'s aug: {}".format(k, amount))
        smote_sample_dict[k] = smote_list

    total_acc = 0
    for c, representation_list in smote_sample_dict.items():
        c += 1  # class mapping
        assert 1 <= c <= class_num
        targets = (np.ones([window_size, window_size], dtype=int) * c)
        for i in range(len(representation_list)):
            new_dataset.data.append(representation_list[i])
            new_dataset.labels.append(targets)
    print("SMOTE-representation aug:{} !!!".format(len(new_dataset)))

    # set dataloader
    smote_dataloader = DataLoader(new_dataset, 100, shuffle=True)

    print_layers = False
    for name, param in network.named_parameters():
        if print_layers: (print(name))
        if name.find("classifier"):
            param.requires_grad = False
            if print_layers: print("{} is frozen!!!".format(name))
        else:
            continue
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=1e-4)

    loss_fn = torch.nn.MSELoss()
    epoch = 60
    predict = torch.zeros([class_num, class_num]).cpu()
    for e in range(epoch):
        count = 0
        for representation, one_hot_targets in smote_dataloader:
            one_hot_targets = one_hot_targets.to(device)
            outputs = network.classifier(representation.to(device))
            if torch.is_complex(outputs):
                loss_real = loss_fn(outputs.real, one_hot_targets.real)
                loss_imag = loss_fn(outputs.imag, one_hot_targets.imag)
                loss = (loss_real + loss_imag) / 2
                distance = abs(outputs - (1 + 1j))
            else:
                assert 0
            prediction = torch.argmin(distance, dim=1).to(device)
            targets = torch.argmax(abs(one_hot_targets), dim=1).to(device)

            # Count the number of predictions for each category
            for i in range(prediction.shape[0]):
                predict[targets[i], prediction[i]] += 1

            acc = (prediction == targets).sum().item()
            total_acc += acc
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            count += 1
            if print_training:
                print("\rTraining with smote features...{:.2%}".format(
                    (count + len(smote_dataloader) * e) / (len(smote_dataloader) * epoch)), end="")
    print("\nTotal smote_accuracy:{:.2%}".format(total_acc / (len(new_dataset) * epoch)))

    for i in range(predict.shape[0]):
        predict[i, :] = predict[i, :] / predict[i, :].sum()

    # Draw heat map:
    plt.figure(dpi=300)
    data = pd.DataFrame(np.array(100 * predict))
    plot = sns.heatmap(data, square=False, annot=True, fmt='.2f', cmap="Wistia",
                       vmax=100.00, vmin=0.00,
                       annot_kws={'size': 6})
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylabel("True label", fontsize=10)
    plt.xlabel("Predicted label", fontsize=10)
    # 然后调整color bar
    cbar = plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    # 加入标题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

    plt.title("WSMOTE{} Figure of training results during further training:\n".format(a))
    plt.show()
    try:
        plt.savefig(save_path + "WSMOTE{}FurtherTraining.png".format(a))
    except:
        print("The training result graph WSMOTE{} for further training is not saved!!!")

    # 保存网络模型
    try:
        torch.save(network, save_path + "WSMOTE_{}.pth".format(a))
    except:
        print("The model is not saved to:")
        print(save_path + "WSMOTE_{}.pth".format(a))
    else:
        print("The model is not saved to:" + save_path + "WSMOTE_{}.pth".format(a))
    print("*" * 30)

    return network