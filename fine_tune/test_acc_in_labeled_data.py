import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def kappa_cal(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)


def get_confusion_matrix_on_whole_lables(net60, full_dataset, path, net_name, t=None, dtype_is_complex=True):
    print("=" * 30)
    print("Verify accuracy for all labeled datasets ({})".format(len(full_dataset)))
    dataloader = DataLoader(full_dataset, 100, shuffle=True)
    print("Load the trained network")

    acc60 = 0
    class_num = full_dataset.__get_class_num__()
    prediction_matrix = np.zeros([class_num, class_num])
    _, ground_truth = full_dataset.__get_whole_image__()

    net60.eval()
    for input, targets, _ in dataloader:
        if dtype_is_complex:
            input = input.cuda()
            output60 = net60(input)
            distance60 = abs(output60 - (1 + 1j))
        else:
            input = input.real
            input = input.cuda()
            output60 = net60(input)
            distance60 = abs(output60 - 1)

        prediction60 = torch.argmin(distance60, dim=1)
        acc60 += (prediction60 == abs(targets.cuda()).argmax(1)).sum().item() / len(full_dataset)
        for prediction, GT in zip(prediction60, torch.argmax(abs(targets), dim=1)):
            prediction_matrix[int(GT), int(prediction)] += 1
    classes_num = np.sum(prediction_matrix, 1, np.int)
    AA = 0
    for i in range(class_num):
        print("class{}: {:.2%} {}/{}".format(i, prediction_matrix[i, i] / np.sum(prediction_matrix, 1)[i],
                                             int(prediction_matrix[i, i]), int(np.sum(prediction_matrix, 1)[i])))
        AA += prediction_matrix[i, i] / np.sum(prediction_matrix, 1)[i]
        for j in range(class_num):
            prediction_matrix[i, j] = prediction_matrix[i, j] / classes_num[i]

    np.savetxt(fname=path + "confusion_matrix_{}.txt".format(t), X=100 * prediction_matrix, fmt="%.2f")

    # Draw heat map:
    plt.figure(dpi=300)
    data = pd.DataFrame(100 * prediction_matrix)
    plot = sns.heatmap(data, square=False, annot=True, fmt='.2f', cmap="Blues",
                       vmax=100.00, vmin=0.00,
                       annot_kws={'size':6},
                       )
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylabel("True label", fontsize=10)
    plt.xlabel("Predicted label", fontsize=10)
    cbar = plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    plt.title(path +"\nconfusion_matrix_{}_{}.png\n OA={:.2%},AA={:.2%},Kappa={:.2%}\n".format(net_name, t, acc60, AA / class_num,
                                                                    kappa_cal(prediction_matrix)), fontsize='small')
    plt.savefig(path + "confusion_matrix_{}.png".format(t))
    plt.show()
    print(
        "=" * 30,
        "\ntrained for 60 epoch:\n"
        "Overall Accuracy:{:.2%}\nAverage Accuracy:{:.2%}".format(acc60, AA / class_num))
    print("=" * 30)
