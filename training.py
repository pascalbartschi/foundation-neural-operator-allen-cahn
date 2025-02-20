import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from data_processing import avg_rel_L2_error


class AllenCahnDataset(Dataset):
    def __init__(self, data, epsilon_values, time_points, simple_testing = False):
        """
        data: dictionary mapping epsilon values to numpy arrays of shape (n_samples, n_timesteps, n_points)
        epsilon_values: list of epsilon values
        time_points: numpy array of time points
        """
        # add values
        self.data = data
        self.epsilon_values = epsilon_values
        self.time_points = time_points

        # initialize dimensions for index calculation
        self.n_samples = len(data[epsilon_values[0]])
        self.n_epsilons = len(epsilon_values)
        self.T = len(time_points)

        # for testing we only want time pairs (0, T-1)
        if simple_testing:
            self.time_pairs = [(0, i) for i in range(1, self.T)]
        else:
            self.time_pairs = [(i, j) for i in range(0, self.T) for j in range(i + 1, self.T)]

        self.len_timepairs = len(self.time_pairs)
        self.combinations_per_epsilon = self.n_samples * self.len_timepairs

     
        # # Create index mapping
        # self.indices = []
        # for eps in epsilon_values:
        #     n_samples = len(data[eps])
        #     self.indices.extend([(eps, i) for i in range(n_samples)])
    
    def __len__(self):
        return self.n_epsilons * self.n_samples * self.len_timepairs
   
    
    def __getitem__(self, idx):
        # calculate indices
        # epsilon: think about in which segment of the data we are
        eps_idx = idx // (self.n_samples * self.len_timepairs)
        # sample: think about which sample we are (//) after accounting for position in epsilon (%)
        sample_idx = (idx % (self.n_samples * self.len_timepairs)) // self.len_timepairs
        # time_pair: simply index at last axis
        time_pair_idx = idx % self.len_timepairs

        # Retrieve time pair and calculate time
        t_inp, t_out = self.time_pairs[time_pair_idx]
        time = self.time_points[t_out] - self.time_points[t_inp]

        # Retrieve epsilon value
        eps = self.epsilon_values[eps_idx]

        # Prepare inputs and outputs
        inputs = self.data[eps][sample_idx, t_inp, :].type(torch.float32) # this input holds no grid
        # inputs = (inputs - self.mean) / self.std  # Normalize
        inputs_t = (torch.ones(inputs.size(0)) * time) # Add time channel
        inputs_g = torch.linspace(-1, 1, inputs.size(0))  # Add grid channel
        inputs_eps = (torch.ones(inputs.size(0)) * eps) # Add epsilon channel
        inputs = torch.stack((inputs, inputs_g, inputs_t, inputs_eps), -1)
        # inputs = torch.cat((inputs, inputs_g, inputs_t), -1)

        outputs = self.data[eps][sample_idx, t_out, :].type(torch.float32).view(-1, 1) # omit grid in outputs
        # outputs = (outputs - self.mean) / self.std  # Normalize

        # return float(eps), float(time), inputs, outputs
    
        return {
            'initial': inputs,
            'target': outputs,
            'epsilon': torch.FloatTensor([eps]),
            'time': torch.FloatTensor([time])
        }



        # eps, sample_idx = self.indices[idx]
        # trajectory = self.data[eps][sample_idx]
        
        # return {
        #     'initial': torch.FloatTensor(trajectory[0]),
        #     'target': torch.FloatTensor(trajectory[1:]),
        #     'epsilon': torch.FloatTensor([eps]),
        #     'times': torch.FloatTensor(self.time_points[1:])
        # }
# def periodic_boundary_loss(u):
#     """
#     Compute periodic boundary condition loss for FNO outputs.
    
#     Args:
#         u (torch.Tensor): Predicted solution with shape (batch_size, height, width, channels).
#         spatial_dim (int): Dimension representing the spatial variable (default: 2 for height).
    
#     Returns:
#         torch.Tensor: Periodic boundary condition loss.
#     """
#     # Match values at the boundaries
#     loss_value = torch.mean((u[:, 0, :] - u[:, -1, :]) ** 2)
    
#     # Match first derivatives at the boundaries
#     grad_left = torch.autograd.grad(u[:, 0, :].sum(), u, create_graph=True)[0]
#     grad_right = torch.autograd.grad(u[:, -1, :].sum(), u, create_graph=True)[0]
#     loss_derivative = torch.mean((grad_left - grad_right) ** 2)
    
#     return loss_value + loss_derivative

def train_model(model, train_data, val_data, epsilon_values, time_points, 
                batch_size=32, epochs=100, device='cuda',
                learning_rate=1e-3, weight_decay = 1e-7, 
                epoch_patience = 8, curriculum_steps=None, beta_=0.1, gamma_=0.1):
    """
    Training loop with curriculum learning on epsilon values.
    
    curriculum_steps: list of (epoch, epsilon_subset) tuples defining when to introduce each epsilon value
    """
    train_dataset = AllenCahnDataset(train_data, epsilon_values, time_points)
    val_dataset = AllenCahnDataset(val_data, epsilon_values, time_points)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epoch_patience)

    l_mse = nn.MSELoss().to(device)
    l_pbc  = lambda u: torch.mean((u[:, 0, :] - u[:, -1, :]) ** 2)
    l_smooth = lambda predictions: ((predictions.squeeze(-1)[:, 1:] - predictions.squeeze(-1)[:, :-1]) ** 2).mean()

    
    model = model.to(device)
    best_val_loss = float('inf')

    # Initialize lists to store loss and accuracy values for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    with tqdm(range(epochs), desc=f"Training") as tbar:
        for epoch in tbar:
            # Update curriculum if needed
            if curriculum_steps:
                for step_epoch, eps_subset in curriculum_steps:
                    if epoch == step_epoch:
                        train_dataset = AllenCahnDataset(train_data, eps_subset, time_points)
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        # reset lr
                        # optimizer = optimizer.param_groups[0]['lr'] = learning_rate
                        print(f"Curriculum update: now training on epsilon values {eps_subset}")
            
            # Training
            model.train()
            train_loss = 0
            train_accuracy = 0
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Forward pass - implement your model to handle these inputs
                pred = model(batch['initial'], batch['epsilon'], batch['time'])
                
                # Compute loss
                loss = l_mse(pred, batch['target']) + beta_ * l_pbc(pred).to(device) + gamma_ * l_smooth(pred).to(device)
                loss.backward()
                optimizer.step()

                # Calculate batch accuracy (mean absolute error as an example)
                batch_accuracy = avg_rel_L2_error(pred, batch['target'])

                # Update loss and accuracy
                train_accuracy += batch_accuracy
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pred = model(batch['initial'], batch['epsilon'], batch['time'])
                    val_loss += (l_mse(pred, batch['target']) + beta_ * l_pbc(pred).to(device)).item()

                    # Calculate batch accuracy (mean absolute error as an example)
                    batch_accuracy = avg_rel_L2_error(pred, batch['target'])
                    # print(batch_accuracy)
                    # print(pred.size())
                    # Update metrics
                    val_accuracy += batch_accuracy

            # print(len(train_loader.dataset))
            
            # normalize and log metrics
            train_loss /= len(train_loader.dataset)
            train_accuracy /= len(train_loader.dataset)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            val_loss /= len(val_loader.dataset)
            val_accuracy /= len(val_loader.dataset)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            tbar.set_postfix(
                train_loss=train_loss, 
                val_loss=val_loss, 
                # train_accuracy=train_accuracy,
                # val_accuracy=val_accuracy, 
                lr=optimizer.param_groups[0]['lr']
                )
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # add hyperparams and val_loss to identify best model in save string
                best_state_dict = model.state_dict()

    # Save best model
    torch.save(best_state_dict, 
                f"best_model_{model.__class__.__name__}_lr{learning_rate}_bs{batch_size}_epochs{epochs}_valloss{best_val_loss:.6f}.pt")
    
    # Plot Training and Validation Loss and Accuracy
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses) #, label='Train Loss')
    plt.plot(range(epochs), val_losses) #, label='Val Loss')
    plt.title('$\ell_2$-Loss')
    plt.xlabel('Epoch')
    plt.ylabel('$\ell_2$-Loss')
    # plt.legend()

    # # Accuracy subplot
    # plt.subplot(1, 2, 2)
    # plt.plot(range(epochs), train_accuracies, label='Train')
    # plt.plot(range(epochs), val_accuracies, label='Val')
    # plt.title('Average Relative $\ell_2$')
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Relative $\ell_2$')
    # plt.legend()

    # plt.tight_layout()
    plt.show()

    # return the trained model
    return model

def test_model(model, test_data, epsilon_values, time_points, show_plot=True):
    # create dataset and loader
    test_dataset = AllenCahnDataset(test_data, epsilon_values, time_points, simple_testing = True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


    device = next(model.parameters()).device

    # Assuming test_loader is a DataLoader object for the test_sol dataset
    test_relative_l2_epsilon = {e: 0.0 for e in epsilon_values}
    test_relative_l2_time = {t: 0.0 for t in time_points[1:]}

    # Test the model
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode

        for batch in test_loader:

            # move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            # predict
            pred = model(batch['initial'], batch['epsilon'], batch['time']) #.squeeze(2)

            for t in test_relative_l2_time.keys():
                time_mask = (batch["time"] == t).squeeze()
                test_relative_l2_time[t] += avg_rel_L2_error(pred[time_mask, :], batch["target"][time_mask, :])

            for e in test_relative_l2_epsilon.keys():
                eps_mask = (batch["epsilon"] == e).squeeze()
                test_relative_l2_epsilon[e] += avg_rel_L2_error(pred[eps_mask, :], batch["target"][eps_mask, :])
            
            # # error seperately calcualted for pairs (0, t)
            # for t, e in zip(test_relative_l2_time.keys(), test_relative_l2_epsilon.keys()):
            #     # mask predictions for individual times
            #     time_mask = (batch["time"] == t).squeeze()
            #     eps_mask = (batch["epsilon"] == e).squeeze()
            #     # extract l2 per time
            #     batch_relative_l2_t = avg_rel_L2_error(pred[time_mask, :], batch["target"][time_mask, :])
            #     test_relative_l2_time[t] += batch_relative_l2_t #/ time_mask.sum().item())
            #     print(t)
            #     # print(batch_relative_l2_t)
            #     # print(pred[time_mask, :].size())
            #     # print(time_mask.sum().item())
            #     # extract l2 per epsilon
            #     batch_relative_l2_e = avg_rel_L2_error(pred[eps_mask, :], batch["target"][eps_mask, :])
            #     test_relative_l2_epsilon[e] += batch_relative_l2_e #/ eps_mask.sum().item())


        # return test_relative_l2

    # test_relative_l2 = report_L2_error_wtime(test_loader, fno) 
    if show_plot:
        # plot the results
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        labels = ["Time", "Epsilons"]
        for i, test_relative_l2 in enumerate([test_relative_l2_time, test_relative_l2_epsilon]):
        # visualize the results
            ax[i].plot(test_relative_l2.keys(), test_relative_l2.values(), "o--", label='Test Relative L2 Error')
            ax[i].set_title(f'Relative L2 Error Over {labels[i]}')
            ax[i].set_xlabel(labels[i])
            ax[i].set_ylabel('Relative L2 Error')
            ax[i].set_xticks(list(test_relative_l2.keys()))

        plt.show()

    return test_relative_l2_time, test_relative_l2_epsilon
# Example curriculum steps
# curriculum_steps = [
#     (0, [0.1]),           # Start with largest epsilon
#     (20, [0.1, 0.05]),    # Add medium epsilon
#     (40, [0.1, 0.05, 0.02])  # Add smallest epsilon
# ]
