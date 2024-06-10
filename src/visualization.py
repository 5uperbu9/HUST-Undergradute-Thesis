import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test_root = './test_results/'
save_root = './plot/'
epoch = 20

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.size'] = 11
color = [np.array((229, 6, 25)) / 255, np.array((0, 102, 177)) / 255, 'g', 'y']


def method_vis():
    method = ['随机采样', '基于信息量', '基于信息量与代表性']

    # accuracy
    fig, ax = plt.subplots(figsize=(5.7, 3))
    plt.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.15)
    ax.set_xlim(0, max_epoch)
    ax.set_xticks(range(0, max_epoch + 1))
    ax.set_ylim(min_acc, max_acc)
    ax.set_yticks(np.array(range(int(min_acc / 0.1), int(max_acc / 0.1) + 1)) * 0.1)

    acc = np.array(pd.read_csv(test_root + path + 'accuracy.csv', header=None, on_bad_lines='error'))
    for t in range(3):
        a = acc[range(t, 60, 3)]
        acc_mean = np.mean(a, axis=0)
        ax.plot(range(max_epoch + 1), acc_mean[:max_epoch + 1], color=color[t], label=method[t], linewidth=2)

    ax.legend(loc="lower right", ncols=1, fancybox=False, edgecolor='black')
    ax.set_xlabel('迭代次数(次)')
    ax.set_ylabel('准确率(%)')
    fig.savefig(save_root + path + 'accuracy.svg', format='svg')
    plt.show()

    # query times
    fig, ax = plt.subplots(figsize=(5.7, 3))
    plt.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.15)
    ax.set_xlim(1, epoch)
    ax.set_xticks(range(1, epoch + 1))
    ax.set_ylim(128, max_time)
    ax.set_yticks(np.array(range(128, max_time + 1, batch_size)))

    times = np.array(pd.read_csv(test_root + path + 'query_times.csv', header=None)).T
    times_mean = np.mean(times, axis=1)
    times_var = np.var(times, axis=1)
    print(times_mean, times_var)

    for t in range(3):
        ax.plot(range(1, epoch + 1), times[t], color=color[t], label=method[t], linewidth=2)

    ax.legend(loc="upper right", ncols=3, fancybox=False, edgecolor='black', columnspacing=0.7, handletextpad=0.2)
    ax.set_xlabel('测试轮数(轮)')
    ax.set_ylabel('查询次数(次)')
    fig.savefig(save_root + path + 'query_times.svg', format='svg')
    plt.show()


def uncertainty_vis():
    # uncertainty
    fig, ax = plt.subplots(figsize=(5.7, 3))
    plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.15)
    ax.set_xlim(0, max_epoch)
    ax.set_xticks(range(0, max_epoch + 1))
    ax.set_ylim(min_acc, max_acc)
    ax.set_yticks(np.array(range(int(min_acc / 0.1), int(max_acc / 0.1) + 1)) * 0.1)
    labels = ['置信区间', '信息熵']

    acc = np.array(pd.read_csv(test_root + path + 'uncertainty.csv', header=None, on_bad_lines='error'))
    for t in range(2):
        ax.plot(range(max_epoch + 1), acc[t][:max_epoch + 1], color=color[t], label=labels[t],
                linewidth=2)

    ax.legend(loc="lower right", ncols=1, fancybox=False, edgecolor='black', columnspacing=0.5, handletextpad=0.2)
    ax.set_xlabel('迭代次数(次)')
    ax.set_ylabel('准确率(%)')
    fig.savefig(save_root + path + 'uncertainty.svg', format='svg')
    plt.show()


def size_vis():
    # size
    fig, ax = plt.subplots(figsize=(5.7, 3.4))
    plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.13)
    ax.set_xlim(0, 320)
    ax.set_xticks(range(0, 320 + 1, 32))
    ax.set_ylim(0.4, 1)
    ax.set_yticks(np.array(range(4, 10)) * 0.1)

    acc = np.array(pd.read_csv(test_root + path + 'batch_size.csv', header=None, on_bad_lines='error'))
    for t in range(4):
        acc_mean = np.mean(acc[range(t, 24, 4)], axis=0)
        ax.plot(range(0, 256 + 1, 8 * (2 ** t)), acc_mean[:int(32 / (2 ** t)) + 1], color=color[t],
                label='batch size=' + str((2 ** t) * 8),
                linewidth=2)

    ax.legend(loc="lower right", ncols=1, fancybox=False, edgecolor='black', columnspacing=0.5, handletextpad=0.2)
    ax.set_xlabel('查询次数(次)')
    ax.set_ylabel('准确率(%)')
    fig.savefig(save_root + path + 'batch size query.svg', format='svg')
    plt.show()

    fig, ax = plt.subplots(figsize=(5.7, 3.4))
    plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.13)
    ax.set_xlim(0, 5)
    ax.set_xticks(range(0, 6))
    ax.set_ylim(0.4, 1)
    ax.set_yticks(np.array(range(4, 10)) * 0.1)

    for t in range(4):
        acc_mean = np.mean(acc[range(t, 24, 4)], axis=0)
        ax.plot(range(max_epoch + 1), acc_mean[:max_epoch + 1], color=color[t],
                label='cdd_size=' + str((2 ** t) * 64), linewidth=2)
        ax.plot(range(6), acc_mean[:6], color=color[t], label='batch size=' + str((2 ** t) * 8),
                linewidth=2)

    ax.legend(loc="lower right", ncols=1, fancybox=False, edgecolor='black', columnspacing=0.5, handletextpad=0.2)
    ax.set_xlabel('迭代次数(次)')
    ax.set_ylabel('准确率(%)')
    fig.savefig(save_root + path + 'batch size iter.svg', format='svg')
    plt.show()


def initial_vis():
    categories = ['dermatology', 'penguins size', 'MNIST', 'tomato leaf disease']

    fig, ax = plt.subplots(figsize=(5.7, 3))
    plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.15)

    bar_width = 0.35
    ax.set_xlabel('数据集')
    ax.set_ylabel('准确率/%')
    ax.set_ylim(0, 1)
    index = np.arange(len(categories))
    ax.set_xticklabels(labels=categories)
    ax.set_xticks(index + bar_width / 2)

    acc = np.array(pd.read_csv(test_root + '/initial.csv', header=None)).T
    bars1 = ax.bar(index, acc[0], bar_width, color=color[0], label='random')
    bars2 = ax.bar(index + bar_width, acc[1], bar_width, color=color[1], label='K-Means++')
    ax.legend(loc="upper right", ncols=1, fancybox=False, edgecolor='black', columnspacing=0.5, handletextpad=0.2)

    for bar in bars1:
        height = bar.get_height() + 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, height, '{height}', ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height() + 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, height, '{height}', ha='center', va='bottom', fontsize=10)

    plt.show()
    fig.savefig(save_root + 'initial seed.svg', format='svg')


if __name__ == '__main__':

    data = 4

    if data == 2:
        path = 'dermatology/'
        batch_size = 12
        max_epoch = 5
        min_acc = 0.55
        max_acc = 1
        max_time = 108
    elif data == 1:
        path = 'penguins/'
        batch_size = 12
        max_epoch = 5
        min_acc = 0.65
        max_acc = 1
        max_time = 102
    elif data == 3:
        path = 'tomato_leaf/'
        batch_size = 512
        max_epoch = 20
        min_acc = 0.2
        max_acc = 0.9
        max_time = 2400
    else:
        path = 'MNIST/'
        batch_size = 64
        max_epoch = 10
        min_acc = 0.3
        max_acc = 1
        max_time = 832

    initial_vis()
    method_vis()
    uncertainty_vis()
    size_vis()
