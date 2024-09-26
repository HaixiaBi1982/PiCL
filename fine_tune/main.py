import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset_preparing.dataset_aug import count_dataset_classes_num, dataset_sampling, dataset_augmentation, rotate90, \
    rotate270, HFlip_rotate90, HFlip_rotate180, \
    HFlip_rotate270, rotate180, HorizontalFlip, dataset_addition, add_complex_noise, PSG_OF_Nearset_Wishart, \
    dataset_augmentation_of_wishartPSG
from dataset_preparing.dataset_def import DataSet
from dataset_preparing.standardization import dataset_standardization
from fine_tune.PLG import count_classnum_and_def_minorityclasses, aug_dataset_by_SLIC
from fine_tune.CV_WSMOTE import SMOTE
from test_acc_in_labeled_data import get_confusion_matrix_on_whole_lables
from network.ComplexAttentionNet import *


# Set parameters
batch_size = 100
learning_rate = 0.002
epoch = 60
window_size = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting Dataset
dataset_name = "FlveoBig"
net_name = 'ComplexAttentionNet'
print("=" * 40 + dataset_name + "=" * 40)
dataset_path = "../benchmark_dataset/" + dataset_name
pretrained_model_path = "../trained_model/" + dataset_name + "/pretrained_model/"
fine_tuned_model_path = "../trained_model/" + dataset_name + "/fine-tuned_model/" + net_name + "/"

sampling_rate = 0.2 / 100
start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
start_time = "sampling_rate_{}_".format(sampling_rate) + start_time

if not os.path.exists(fine_tuned_model_path + start_time):
    os.makedirs(fine_tuned_model_path + start_time)
else:
    assert 0, "The dir path is already existed."
fine_tuned_model_path = fine_tuned_model_path + start_time + "/"

full_dataset = DataSet(dataset_path=dataset_path,
                       window_size=window_size,
                       stride=5,
                       std=False,
                       print_show=True,
                       return_position=True)
mean, std = full_dataset.__get_mean_and_std__()
class_num = full_dataset.__get_class_num__()

# 数据集采样
print("\nDataset with labels:")
count_dataset_classes_num(full_dataset)

sampled_dataset = dataset_sampling(full_dataset, sampling_rate) 
class_ratio_in_dataset = count_dataset_classes_num(sampled_dataset)

epoch60_acc = []
# ======================================
PLG = True
PSG = True
CV_WSMOTE = True
# ======================================

if PLG:
    class_ratio_in_dataset = np.array(class_ratio_in_dataset) / len(sampled_dataset)
    reciprocal_arr = 1 / class_ratio_in_dataset
    thresholds_of_PLG_in_each_class = (reciprocal_arr / reciprocal_arr.sum()) * len(sampled_dataset)
    print(f"thresholds_of_PLG_in_each_class = {thresholds_of_PLG_in_each_class}")

    SLIC_dataset = aug_dataset_by_SLIC(dataset=sampled_dataset,
                                       classes_list=count_classnum_and_def_minorityclasses(dataset=sampled_dataset),
                                       n_segments_size=5,
                                       show_img=False,
                                       thresholds_of_PLG_in_each_class=thresholds_of_PLG_in_each_class)
if PSG:
    # Noise PSG
    AddPhysicalNoise = add_complex_noise(scale=0.15)
    # Weak PSG
    aug_list1 = [AddPhysicalNoise, rotate90, rotate180, rotate270, HorizontalFlip, HFlip_rotate90, HFlip_rotate180,
                 HFlip_rotate270]
    aug_dataset = dataset_augmentation(dataset=sampled_dataset, augment_list=aug_list1)

    # Wishart PSG
    psg_of_wishart = PSG_OF_Nearset_Wishart(dataset=sampled_dataset,
                                            random_select=True,
                                            number_of_selected_pixels=25,)
    wishartPSG_aug_dataset = dataset_augmentation_of_wishartPSG(dataset=sampled_dataset, augment_list=[psg_of_wishart])
    print("The wishartPSG_augmented_dataset:")
    count_dataset_classes_num(wishartPSG_aug_dataset)

# Finally Training Dataset
dataset = sampled_dataset
str_list = []
if PLG:
    dataset = dataset_addition(sampled_dataset, SLIC_dataset)
    str_list.append("SLIC")
if PSG:
    dataset = dataset_addition(dataset, aug_dataset)
    dataset = dataset_addition(dataset, wishartPSG_aug_dataset)
    str_list.append("WishartPSG")
    str_list.append("augmentation")
if CV_WSMOTE:
    str_list.append("wsmote")
strs = '_'.join(str_list)
# Create dir path
print("dataset = sampled_dataset + " + ' + '.join(str_list))
if (not os.path.exists(fine_tuned_model_path + "/" + strs)) and (len(str_list) != 0):
    os.makedirs(fine_tuned_model_path + "/" + strs)
    fine_tuned_model_path = fine_tuned_model_path + strs + "/"
dataset = dataset_standardization(dataset, torch.as_tensor(mean), torch.as_tensor(std))

train_data_size = len(dataset)
print("Finally in Training Dataset:")
count_dataset_classes_num(dataset)
train_dataloader = DataLoader(dataset, batch_size, shuffle=True)

class CMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, labels):
        # L2 loss
        loss_a = self.mse_loss(inputs.real, labels.real)
        loss_b = self.mse_loss(inputs.imag, labels.imag)
        loss = 0.5 * (loss_a + loss_b)
        return loss.mean()

# training for a times
for a in range(1, 10):

    print("-----------------{} 训练轮数：{}-----------------".format(net_name, a))
    loss_fn = CMSELoss()
    loss_fn = loss_fn.to(device)
    '''ComplexAttentionNet'''
    byol_net = torch.load(pretrained_model_path + "{}/{}.pth".format(net_name, net_name))
    net = byol_net.online_encoder.net
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)

    total_train_step = 0
    total_test_step = 0
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    epoch_num = []

    acc1, acc2, acc3 = 0, 0, 0
    for i in range(epoch):
        net.train()
        total_loss = 0
        total_acc = 0
        epoch_batches = 0
        for data in train_dataloader:
            imgs, targets, _ = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            distance = abs(outputs - (1 + 1j))

            total_loss += loss.item()
            prediction = torch.argmin(distance, dim=1)
            acc = (prediction == abs(targets).argmax(1)).sum().item()
            total_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_batches = epoch_batches + 1

        epoch_num.append(i + 1)
        train_loss.append(total_loss / epoch_batches)
        train_acc.append(total_acc / train_data_size)

        if (i + 1) == epoch:
            print("time：" + datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
            print("Epoch:{} Train acc：{:.2%}".format(i + 1, total_acc / train_data_size))

    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1), plt.plot(epoch_num, train_loss, ".-"), plt.xlabel("Epoches"), plt.ylabel(
        "Train Loss"), plt.title(
        "Train Loss")
    plt.subplot(2, 2, 2), plt.plot(epoch_num, train_acc, ".-"), plt.xlabel("Epoches"), plt.ylabel(
        "Train Acc"), plt.title(
        "Train Accuracy")
    plt.suptitle("{}".format(net_name))
    plt.show()

    get_confusion_matrix_on_whole_lables(net60=net,
                                         t=a,
                                         full_dataset=dataset_standardization(full_dataset, mean, std),
                                         dtype_is_complex=True,
                                         path=fine_tuned_model_path,
                                         net_name=net_name)
    torch.save(net, fine_tuned_model_path + "{}_{}.pth".format(net_name, a))
    print("network '{}_{}.pth' is saved".format(net_name, a))

    #  ===============================================================================================================
    if CV_WSMOTE:
        if PSG:
            s_a_dataset = dataset_addition(sampled_dataset, aug_dataset)
        else:
            s_a_dataset = sampled_dataset
        print("CV-WSMOTE to the following dataset:")
        count_dataset_classes_num(s_a_dataset)

        network = SMOTE(original_dataset=
                        dataset_standardization(s_a_dataset, torch.from_numpy(mean), torch.from_numpy(std)),
                        network=net,
                        net_name=net_name,
                        a=a,
                        window_size=window_size,
                        save_path=fine_tuned_model_path,
                        print_training=False)
        get_confusion_matrix_on_whole_lables(network,
                                             dataset_standardization(full_dataset, mean, std),
                                             t="WSMOTE" + str(a),
                                             path=fine_tuned_model_path,
                                             net_name=net_name)
print("Fine-tuning Finished!")
