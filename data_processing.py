import numpy as np
import matplotlib.pyplot as plt
import torch

def load_npz_data(file_path, for_exploration = False):

    # load the data npz file standard traingin data
    all_data = np.load(file_path)
    # split off the espilon keys again,  e.g. "fourier_0.1" to two level dict with "fourer" and "0.1"
    split_data = {}
    for key in all_data.keys():
        method, epsilon = key.split("_")
        if method not in split_data.keys():
            split_data[method] = {}
        split_data[method][float(epsilon)] = torch.Tensor(all_data[key])


    if for_exploration:
        return split_data
    else:
        # create a dictionary where epsilon are the keys and the values are the mixed samples of the corresponding ICs
        mixed_dict = {}
        for IC in split_data.keys():
            for epsilon in split_data[IC].keys():
                if epsilon not in mixed_dict:
                    mixed_dict[epsilon] = []
                mixed_dict[epsilon].append(split_data[IC][epsilon])

        # print(len(mixed_dict[0.02]))

        # stack the mixed samples and randomly shuffle them
        for epsilon in mixed_dict.keys():
            mixed_dict[epsilon] = torch.cat(mixed_dict[epsilon], dim=0)
            # print(mixed_dict[epsilon].shape)
            idx = torch.randperm(mixed_dict[epsilon].shape[0])
            mixed_dict[epsilon] = mixed_dict[epsilon][idx]

        return mixed_dict

def visualize_sample(data, ic = "gmm", epsilons = None, sample_id = None, set_title = True):

    if ic not in data.keys():
        raise ValueError(f"IC {ic} not in data keys")
    
    epsilons = list(data[ic].keys()) if epsilons is None else epsilons

    # data = all_data[f"{ic}_{eps}"].copy()

    fig, axs = plt.subplots(1, len(epsilons), figsize=(15, 5), sharey=True)

    domain = [-1, 1]
    sample_id = np.random.randint(data[ic][epsilons[0]].shape[0]) if sample_id is None else sample_id
    labels = [f"$u(t = {i/4 * 1e-2})$" for i in range(5)]
    colors = ["#B0BEC5", "#81D4FA", "#29B6F6", "#0288D1", "#01579B"]# ["#E0F7FA", "#81D4FA", "#29B6F6", "#0288D1", "#01579B"]
# ["#006400", "#8FBC8F", "#FFD700", "#FF8C00", "#8B0000"]# ['#0000FF', '#4040BF', '#808080', '#BFBFBF', '#FFBFBF']

    for i, eps in enumerate(epsilons):
        plot_data = data[ic][eps][sample_id, :, :].clone() # make code more readable
        # print(plot_data.shape)
        for j in range(5):
            axs[i].plot(np.linspace(domain[0], domain[1], plot_data[j, :].shape[-1]), plot_data[j, :],  "-", label = labels[j], color = colors[j], alpha = 1, markersize= 1)
            # axs[i].scatter(np.linspace(domain[0], domain[1], plot_data[j, :].shape[-1]), plot_data[j, :], color = colors[j], s= 1)
            axs[i].grid(True, which="both", ls=":")
            axs[i].set_title(f"$\epsilon$ = {eps}")
            axs[i].set_xlabel("$x$")
    
    axs[0].set_ylabel("$u(x,t)$")
    axs[-1].legend(loc = "lower right")

    if set_title:
        fig.suptitle(f"Sample {sample_id} from {ic} initial condition with different $\epsilon$")
    fig.tight_layout()
    plt.show()

def avg_rel_L2_error(u, u_true):
    """
    Compute the average relative L2 error between the true solution and the predicted solution.
    """

    return torch.mean(torch.norm(u - u_true, dim=(-2, -1)) / (torch.norm(u_true, dim=(-2, -1)) + 1e-8)).item()