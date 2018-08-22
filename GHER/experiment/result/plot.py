import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse
import matplotlib.font_manager as fm

def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))     # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')      # 卷积平滑
    return xsmoo, ysmoo

def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None

    # keys 包括 ['test/episode', 'stats_o/std', 'stats_g/std', 'epoch', 'test/success_rate', 'stats_g/mean', 
    #           'stats_o/mean', 'train/success_rate', 'train/episode', 'test/mean_Q']

    keys = [name.strip() for name in lines[0].split(',')]                        # 第一行存储名称
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)  # data.shape=(n_epochs, 10)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    """
        根据需要将数据长度对其至maxlen
    """
    maxlen = np.max([len(x) for x in xs])   # maxlen=50, len(xs)=1, xs[0].shape=(50,)  其中50为周期数
    
    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)   # len(padded_xs)=1, padded_xs[0].shape=(50,)
       
        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value  # 根据需要将数据长度对其至maxlen

        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="./")
parser.add_argument('--smooth', type=int, default=1)
args = parser.parse_args()

# Load all data. 在目录下搜索 progress.csv, 找到后转换为 绝对路径
data = {}
paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'progress.csv'))]

for curr_path in paths:
    if not os.path.isdir(curr_path):
        continue
    # 调用函数读入文件, results是一个字典
    results = load_results(os.path.join(curr_path, 'progress.csv'))
    if not results:
        print('skipping {}'.format(curr_path))
        continue
    print('loading {} ({})'.format(curr_path, len(results['epoch'])))
    with open(os.path.join(curr_path, 'params.json'), 'r') as f:
        params = json.load(f)

    success_rate = np.array(results['test/success_rate'])
    epoch = np.array(results['epoch']) + 1
    env_id = params['env_name']
    replay_strategy = params['replay_strategy']

    if replay_strategy == 'future':
        config = 'HER'
    elif replay_strategy == "none":
        config = 'DDPG'
    else:
        li = replay_strategy.split("-")[:-1]
        config = "-".join(li) + "-her"

    if 'Dense' in env_id:
        config += '-dense'
    else:
        config += '-sparse'
    env_id = env_id.replace('Dense', '')

    # Process and smooth data. 提取周期和成功率,平滑后绘图
    assert success_rate.shape == epoch.shape
    x = epoch
    y = success_rate

    if args.smooth:
        x, y = smooth_reward_curve(epoch, success_rate)
    # assert x.shape == y.shape

    if env_id not in data:
        data[env_id] = {}
    if config not in data[env_id]:
        data[env_id][config] = []
    data[env_id][config].append((x, y))

# print("\nEpiaosde:\n", data['FetchReach-v1']['her-sparse'][0][0])    # 周期计数
# print("\nSuccess_rate:\n", data['FetchReach-v1']['her-sparse'][0][1])    # 成功率


myfont = fm.FontProperties(size=20)
# Plot data.
for env_id in sorted(data.keys()):
    print('exporting {}'.format(env_id))
    #plt.clf()
    plot_clip = 1000
    if env_id == "FetchSlide-v1":
        plot_clip = 1000
    if env_id == "FetchReach-v1":
        plot_clip = 100
    if env_id == "FetchPush-v1":
        plot_clip = 400
    if env_id == "FetchPickAndPlace-v1":
        plot_clip = 600

    plt.figure(figsize=(15, 7))

    ind = 0
    ind1 = 0
    config_list = []
    for config in ["G-HER-her-sparse", "HER-sparse", "DDPG-sparse", "DDPG-dense"]:
        print(config)
        config_list.append(config)

        xs, ys = zip(*data[env_id][config])
        xs, ys = pad(xs), pad(ys)            # pad之前xs[0].shape=(50,), pad之后,xs.shape=(2, 50)
        assert xs.shape == ys.shape
        # print("config =",config)
        if config not in ["DDPG-sparse", "HER-sparse", "DDPG-dense"]:
            c = ["red", "black", "m", "yellow", "green", "blue", "darkslategrey"][ind1]
            ind1 += 1
            # c = "r"
        else:
            c = ["green", "blue", "darkslategrey"][ind]
            ind += 1 

        # ys 按行求取中位数
        if plot_clip != 1000:
            plt.plot(xs[0][:plot_clip], np.nanmedian(ys, axis=0)[:plot_clip], color=c, lw=2)
            # 按行百分数位数, 并填充区域
            plt.fill_between(xs[0][:plot_clip], np.nanpercentile(ys, 25, axis=0)[:plot_clip], np.nanpercentile(ys, 75, axis=0)[:plot_clip], alpha=0.25, color=c)
        else:
            plt.plot(xs[0], np.nanmedian(ys, axis=0), color=c)
            # plt.plot(xs[0], np.nanmedian(ys, axis=0), color=c)
            plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=c)
    
    plt.legend([k.replace("-her-sparse", "-sparse") for k in config_list], prop=myfont, loc=0)
    
    plt.title(env_id)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Median Success Rate', fontsize=16)
    plt.legend()
    # plt.savefig(os.path.join("pic", env_id+".pdf"))
    plt.show()
