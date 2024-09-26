import datetime
import os
import sys
from torch import nn

sys.path.append("..")

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset_preparing.dataset_def import DataSet
from network.ComplexAttentionNet import ComplexAttentionNet
from byol_pytorch.complex_byol import ComplexBYOL

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device("cuda")

import numpy as np
import pandas as pd
def write_array_log_to_excel(array, path, file_name, sheet_name):
    data = pd.DataFrame(np.array(array))
    writer = pd.ExcelWriter(path + '{}.xlsx'.format(file_name))
    data.to_excel(writer, sheet_name, float_format='%.5f')
    writer.save()
    writer.close()


# constants
BATCH_SIZE = 512
MAX_EPOCHS = 200
LR = 0.001
IMAGE_SIZE = 12

# main
if __name__ == '__main__':

    dataset_name = "FlveoBig"
    print("=" * 40 + dataset_name + "=" * 40)

    # FlveoBig: 167712/768000
    dataset_path = "../benchmark_dataset/" + dataset_name

    original_dataset = DataSet(dataset_path=dataset_path,
                           window_size=IMAGE_SIZE,
                           stride=10,
                           std=False,
                           print_show=False,
                           delete_0=False,
                           return_position=True)

    full_dataset = DataSet(dataset_path=dataset_path,
                           window_size=IMAGE_SIZE,
                           stride=10,
                           std=True,
                           print_show=False,
                           delete_0=False,
                           return_position=True)
    mean, std = full_dataset.__get_mean_and_std__()
    class_num = full_dataset.__get_class_num__()

    # define network type:
    net = ComplexAttentionNet(class_num)
    net_name = 'ComplexAttentionNet'
    print("Pretraining ComplexAttentionNet:")
    net = nn.DataParallel(net).cuda()
    pretrained_model_path = "../trained_model/" + dataset_name + "/pretrained_model/" + net_name + "/"

    byol = ComplexBYOL(
        net.module,
        image_size=IMAGE_SIZE,
        hidden_layer=-2,
        projection_size=12,
        projection_hidden_size=108,
        moving_average_decay=0.99,
        mean=mean,
        std=std,
        dataset=original_dataset,
    )

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Pretrained benchmark_dataset length：{}".format(len(full_dataset)))

    optimizer = torch.optim.Adam(byol.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=0)

    train_loss = []
    epoch_num = []

    for i in range(MAX_EPOCHS):
        print("-----------------Epoch {}/{}-----------------".format(i + 1, MAX_EPOCHS))
        byol.train()
        total_loss = 0
        epoch_batches = 0
        for imgs, _, position in train_loader:
            imgs = imgs.to(device)
            loss = byol(imgs, position)

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_batches = epoch_batches + 1

        epoch_num.append(i + 1)
        train_loss.append(total_loss / epoch_batches)

        print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
        print("{}: {} Pretrain Loss = {}".format(dataset_name, net_name, total_loss / epoch_batches))
        torch.save(byol, pretrained_model_path + "{}.pth".format(net_name))

    fig = plt.plot(epoch_num, train_loss, ".-")
    plt.xlabel("Epoches")
    plt.ylabel("Pretrain Loss")
    plt.title("{}: {} Pretrain Loss".format(dataset_name, net_name))
    plt.savefig(pretrained_model_path + "{}_pretrain_loss.png".format(net_name))
    # Save loss curve to excel
    write_array_log_to_excel(array=train_loss, path=pretrained_model_path,
                                                file_name="{}_pretrain_loss.png".format(net_name), sheet_name=net_name)
    plt.show()
    print("Pretraining Finished")


