
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def plot_distributions(MSEs_unseen_by_num_source, params, output_dir):

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

        for npp_idx, npp in enumerate(num_per_perm_list_train):
            for avg_idx, avg_name in enumerate(avg_names[num_source]):
                for model_idx, model_name in enumerate(model_names):
                    x_unseen = (np.arange(MSEs_unseen_mean.shape[2])+1) / (MSEs_unseen_mean.shape[2]+1)

                    fig, ax = plt.subplots()
                    plt.plot(x_unseen, MSEs_unseen_mean[:, npp_idx, :, avg_idx, model_idx].transpose())

                    for distribution_idx, distr in enumerate(distributions):
                        plt.fill_between(x_unseen, MSEs_unseen_min[distribution_idx, npp_idx, :, avg_idx, model_idx], MSEs_unseen_max[distribution_idx, npp_idx, :, avg_idx, model_idx], alpha=0.1)
                    # plt.xticks(x_seen)
                    ax.set_title('Model=' + model_name + ', NumSource=' + str(num_source) + ', NPP=' + str(npp) + ', Avg Func=' + avg_name)
                    ax.legend(distributions)
                    ax.set_xlabel('Percentage of Seen Data')
                    ax.set_ylabel('MSEs avg')
                    ax.xaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
                    ax.set_xticks(np.linspace(0, 1, num=11))

                    plt.savefig(output_dir + model_name + '-' + 'NumSource=' + str(num_source) + ' NPP=' + str(npp) + ' Avg Func=' + str(avg_name) + '.png')

                    plt.close()