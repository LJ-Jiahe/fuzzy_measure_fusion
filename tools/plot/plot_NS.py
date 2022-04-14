from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def plot_num_sources(MSEs_unseen_by_num_source, params, output_dir, axis_left=0.15, axis_right=0.95, y_limit=[-0.08/20, 0.08]):

    for key in params.keys():
        globals()[key] = params[key]

    

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=axis_left, right=axis_right)

    for num_source in num_source_list:
        # MSEs_seen = MSEs_seen_by_num_source[num_source]
        # MSEs_seen_mean = np.mean(MSEs_seen, 0)
        # MSEs_seen_max = np.max(MSEs_seen, 0)
        # MSEs_seen_min = np.min(MSEs_seen, 0)

        MSEs_unseen = MSEs_unseen_by_num_source[num_source]
        MSEs_unseen_mean = np.mean(MSEs_unseen, 0)
        MSEs_unseen_max = np.max(MSEs_unseen, 0)
        MSEs_unseen_min = np.min(MSEs_unseen, 0)


        MSEs = MSEs_unseen
        MSEs_mean = MSEs_unseen_mean
        MSEs_max = MSEs_unseen_max
        MSEs_min = MSEs_unseen_min
        x = (np.arange(MSEs_mean.shape[2])+1) / (MSEs_mean.shape[2]+1)

        
        for distribution_idx, distr in enumerate(distributions):
            for npp_idx, npp in enumerate(num_per_perm_list_train):
                for model_idx, model_name in enumerate(model_names):
                    for avg_idx, _ in enumerate(avg_names[num_source]):

                        if distr == 'uniform' and npp == 5 and model_name =='QP' and avg_idx == 0:
                        
                            plt.plot(x, MSEs_mean[distribution_idx, npp_idx, :, avg_idx, model_idx])

                            plt.fill_between(x, MSEs_min[distribution_idx, npp_idx, :, avg_idx, model_idx], MSEs_max[distribution_idx, npp_idx, :, avg_idx, model_idx], alpha=0.1)


    ax.set_title('Model=QP, Distribution=uniform,\nSample Per Sort=5, Operator=Min')
    plt.legend(num_source_list)
    ax.set_xlabel('Percentage of Observed Sorts')
    ax.set_ylabel('MSE')
    # ax.set_ylim(y_limit)
    ax.xaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    ax.set_xticks(np.linspace(0, 1, num=11))

    plt.savefig(output_dir + 'QP-uniform-NPP=5.png')
    plt.close()



                    
