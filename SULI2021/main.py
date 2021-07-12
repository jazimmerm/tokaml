# from random_tools.DataAnalysis import *
import torch
import pickle
import scipy.io as spio
from random_tools.DataPrep import *
from models.skl_linear import *
from random_tools.tools import *
import models.torch_linear as tl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


if __name__ == '__main__':

    '''
        Shots: 
         [174828.]
         [174829.]
         [174830.]
         [174833.]
         [174860.]
         [174870.]
    '''

    # data = pd.read_parquet('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/all.pq', engine='pyarrow')
    # directory = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
    # sh = DataPrep(174828, directory)
    # fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    # fig.subplots_adjust(hspace=0)
    # ax1, elmdf = sh.split(plot=True, ax=ax1)
    # ax1.set_title(None)
    # xticks = ax1.get_xticks()
    # plot_split_only(sh, ax2)
    # fig.suptitle(f'Raw and Filtered Spectrogram for Shot {sh.shot_no}', y=0.93, fontsize=20)
    # # ax2.set_xticks(xticks)
    # # plt.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(1,1)
    # plot_split_only(sh, ax)
    # plt.show()
    # exit()
    # shots = sorted(os.listdir(directory))
    # shots = [i.split('.')[0] for i in shots if '153' not in i]
    # all_dict = {}
    # for shot in shots[21:22]:
    #     sh = DataPrep(shot, directory)
    #     split_dict = plot_split(sh)
    #     all_dict[shot] = split_dict
    #
    # spio.savemat('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/split_spectrograms.mat', all_dict)
    # exit()
    # lin_model(170891)
    # exit()

    # shots = sorted(os.listdir(directory))
    # shots = [i.split('.')[0] for i in shots if '153' not in i]
    # lenlist = []
    # for shot in shots:
    #     sh = DataPrep(shot, directory)
    #     split = sh.split()
    #     for _, window in split.groupby(level=0):
    #         tim = window.index.get_level_values(level='t_to_elm')[0]
    #         print(tim)
    #         lenlist.append(tim)
    # exit()

    inputDim = 4  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    hiddenDim = 16
    numLayers = 1
    learningRate = 0.0001
    epochs = 100
    batch_size = 128
    shot = 174849
    directory = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'

    # file = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/all.pq'
    # log_file = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/all_log.pq'
    #
    # model = tl.makeModel(inputDim, outputDim, hiddenDim, numLayers, batch_size, epochs, learningRate)
    # model.train_test_split(file)
    #
    # log_model = tl.makeModel(inputDim, outputDim, hiddenDim, numLayers, batch_size, epochs, learningRate)
    # log_model.train_test_split(log_file)
    #
    # lin_model, lin_loss = model.train_linear()
    # log_mod, log_loss = log_model.train_linear()

    lin_loss = np.load('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/models/saved_models/100epochs_loss.npy')
    log_loss = np.load('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/models/saved_models/100epochs_loss_logt.npy')

    log_loss = [100*x for x in log_loss]
    # log_loss = (log_loss-np.min(log_loss))/(np.max(log_loss) - np.min(log_loss))
    # lin_loss = (lin_loss-np.min(lin_loss))/(np.max(lin_loss) - np.min(lin_loss))
    print(log_loss[-10:], lin_loss[-10:])
    exit()
    fig, ax = plt.subplots(1,1)
    ax.plot(range(len(lin_loss)), lin_loss, c='b', label=r'Trained on $t_{ELM}$')
    ax.plot(range(len(log_loss)), log_loss, c='r', label=r'Trained on $log(t_{ELM})$')
    ax.set_ylabel('Mean Squared Error Loss')
    ax.set_xlabel('Epoch')
    ax.set_title(r'Normalized Loss Curves for Models Trained on $t_{ELM}$ and $log(t_{ELM})$')
    plt.legend()
    plt.show()
    exit()
    np.save('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/models/saved_models/100epochs_loss.pt', lin_loss)
    np.save('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/models/saved_models/100epochs_loss_logt.pt', log_loss)
    exit()
    model.test(lin_model, model.test_set)

    lin_model = torch.load('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/models/saved_models/50epochs_log.pt', map_location=torch.device('cpu'))
    # model_df = model.run(lin_model, shot)
    sh = DataPrep(shot, directory)

    new_df = model_df[['t_to_elm', 't_pred']]
    time_unique = new_df.index.unique(level=2).values
    ave_pred = []
    pred_time = []
    for time in time_unique:
        single_time = new_df.xs(time, level=2)
        av = np.mean(single_time['t_pred'].values)
        ave_pred.append(10**av)
        pred_time.append(10**av + time)

        # SCATTER PREDICTIONS ON HEATMAP
    pred_idx = []
    for t in pred_time:
        idx = index_match(time_unique, t)
        pred_idx.append(idx)


    print(r2_score(new_df[~new_df.index.duplicated(keep='first')]['t_to_elm'].values, ave_pred))
    exit()
    # ax1.scatter(pred_idx[::10], np.full_like(pred_idx[::10], 0.5e6), alpha=0.3)
    # plt.show()
    # exit()

    fig, ax1, ax2 = plot_t_to_elm(sh)
    ax2.set_xlabel('Time (ms)', fontsize=14)
    ax2.plot(ave_pred, label='Predicted time to ELM')
    plt.show()







