
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def plot_NPP(MSEs_unseen_by_num_source, params, output_dir, axis_left=0.15, axis_right=0.95, y_limit=[-0.009/20, 0.009]):

    for key in params.keys():
        globals()[key] = params[key]


    for num_source in num_source_list:
        # MSEs_seen = MSEs_seen_by_num_source[num_source]
        # MSEs_seen_mean = np.mean(MSEs_seen, 0)
        # MSEs_seen_max = np.max(MSEs_seen, 0)
        # MSEs_seen_min = np.min(MSEs_seen, 0)

        MSEs_unseen = MSEs_unseen_by_num_source[num_source]
        MSEs_unseen_mean = np.mean(MSEs_unseen, 0)
        MSEs_unseen_max = np.max(MSEs_unseen, 0)
        MSEs_unseen_min = np.min(MSEs_unseen, 0)

        # num_perm = math.factorial(num_source)

        for distribution_idx, distr in enumerate(distributions):
            for avg_idx, avg_name in enumerate(avg_names[num_source]):
                for model_idx, model_name in enumerate(model_names):
                    x_unseen = (np.arange(MSEs_unseen_mean.shape[2])+1) / (MSEs_unseen_mean.shape[2]+1)

                    fig, ax = plt.subplots()
                    plt.subplots_adjust(left=axis_left, right=axis_right)
                    plt.plot(x_unseen, MSEs_unseen_mean[distribution_idx, :, :, avg_idx, model_idx].transpose())

                    for npp_idx, npp in enumerate(num_per_perm_list_train):
                        plt.fill_between(x_unseen, MSEs_unseen_min[distribution_idx, npp_idx, :, avg_idx, model_idx], MSEs_unseen_max[distribution_idx, npp_idx, :, avg_idx, model_idx], alpha=0.1)
                    # plt.xticks(x_seen)
                    ax.set_title('Model=' + model_name + ', Distribution=' + distr + ',\nNum of Source=' + str(num_source) + ', Operator=' + avg_name)
                    ax.legend(num_per_perm_list_train)
                    ax.set_xlabel('Percentage of Observed Permutations')
                    ax.set_ylabel('MSE')
                    # ax.set_ylim(y_limit)
                    ax.xaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
                    ax.set_xticks(np.linspace(0, 1, num=11))

                    plt.savefig(output_dir + 'NPP ' + model_name + '-' +distr + '-' + 'NumSource=' + str(num_source) + ' Operator=' + str(avg_name) + '.png')

                    plt.close()