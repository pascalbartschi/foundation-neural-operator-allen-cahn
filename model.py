import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralConv1d(nn.Module):
    """1D Fourier layer"""
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.weights = nn.Parameter(
            torch.complex(
                torch.randn(in_channels, out_channels, self.modes) / in_channels**0.5,
                torch.randn(in_channels, out_channels, self.modes) / in_channels**0.5
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,
                           device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :self.modes],
            self.weights
        )

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNOBlock(nn.Module):
    def __init__(self, modes, width, use_bn = True):
        super().__init__()

        self.modes = modes
        self.width = width

        # Spectral convolution
        self.spectral_conv = SpectralConv1d(width, width, modes)

        # Pointwise convolution
        self.pointwise_conv = nn.Conv1d(width, width, kernel_size=1)

        # Optional normalization layer (can improve training stability)
        self.norm = nn.BatchNorm1d(width) if use_bn else nn.Identity()
        
    def forward(self, x):

        # Add skip connection betwween spectral and pointwise convoltion
        out = self.spectral_conv(x) + self.pointwise_conv(x)

        # Normalize (optional)
        return self.norm(out)
    
class TCBNEmbedding(nn.Module):
    def __init__(self, channels, use_bn=True):
        super().__init__()
        self.channels = channels

        self.inp2scale = nn.Linear(in_features=1, out_features=channels, bias=True)
        self.inp2bias = nn.Linear(in_features=1, out_features=channels, bias=True)

        self.inp2scale.weight.data.fill_(0)
        self.inp2scale.bias.data.fill_(1)
        self.inp2bias.weight.data.fill_(0)
        self.inp2bias.bias.data.fill_(0)

        
        self.norm = nn.BatchNorm1d(self.channels) if use_bn else nn.Identity()


    def forward(self, x, time):

        x = self.norm(x)
        time = time.reshape(-1,1).type_as(x)
        scale = self.inp2scale(time)
        bias = self.inp2bias(time)
        scale = scale.unsqueeze(2).expand_as(x)
        bias  = bias.unsqueeze(2).expand_as(x)

        return x * scale + bias

# class AllenCahnFNO(nn.Module):
#     def __init__(
#             self, 
#             modes=16, 
#             width=64, 
#             input_channels = 4 # u(x), x, t, eps
#             ):
#         super().__init__()
#         self.modes = modes
#         self.width = width
        
#         # TODO: Initialize model components
#         # Consider:
#         # - Epsilon embedding
#         # - Time embedding
#         # - Input/output layers
#         # - FNO blocks
#         self.width = width

#         self.linear_p = nn.Linear(input_channels, self.width)  # batchsize, N_gridpoints, input_channels [u(x), e, x]

#         # instantiate fourier layers
#         self.fourier_layer1 = FNOBlock(self.modes, self.width)
#         self.fourier_layer2 = FNOBlock(self.modes, self.width)
#         self.fourier_layer3 = FNOBlock(self.modes, self.width)
#         # self.fourier_layer4 = FNOBlock(self.modes, self.width)

#         # time embeddings
#         self.time_layer1 = TCBNEmbedding(self.width)
#         self.time_layer2 = TCBNEmbedding(self.width)
#         self.time_layer3 = TCBNEmbedding(self.width)

#         # epsilon embeddings
#         self.eps_layer1 = TCBNEmbedding(self.width)
#         self.eps_layer2 = TCBNEmbedding(self.width)
#         self.eps_layer3 = TCBNEmbedding(self.width)

#         self.linear_q = nn.Linear(self.width, 32)
#         self.output_layer = nn.Linear(32, 1)

#         self.activation = torch.nn.Tanh()
        

#     def forward(self, x, eps, t):
#         """
#         Args:
#             x: Initial condition (batch_size, x_size)
#             eps: Epsilon values (batch_size, 1)
#             t: Time points (batch_size, n_steps)
#         Returns:
#             Predictions at requested timepoints (batch_size, n_steps, x_size)
#         """
#         # TODO: Implement the full model forward pass
#         # 1. Embed epsilon and time
#         # 2. Process spatial information with FNO blocks
#         # 3. Generate predictions for each timestep

#         x = self.linear_p(x)
#         x = x.permute(0, 2, 1) # to: batchsize, features, gridpoints

#         # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        
#         # layer 1
#         x_space = self.activation(self.fourier_layer1(x))
#         x_time = self.time_layer1(x, t)
#         x_eps = self.eps_layer1(x, eps)
#         x = x_space + x_time + x_eps
#         # layer 2
#         x_space = self.activation(self.fourier_layer2(x))
#         x_time = self.time_layer2(x, t)
#         x_eps = self.eps_layer2(x, eps)
#         x = x_space + x_time + x_eps
#         # layer 3
#         x_space = self.activation(self.fourier_layer3(x))
#         x_time = self.time_layer3(x, t)
#         x_eps = self.eps_layer3(x, eps)
#         x = x_space + x_time + x_eps


#         # x = x[..., :-self.padding]  # pad the domain if input is non-periodic
#         x = x.permute(0, 2, 1)

#         x = self.linear_q(x) # to: batchsize, gridpoints, features
#         x = self.output_layer(x)
        
#         return x

class AllenCahnFNO(nn.Module):
    def __init__(
            self, 
            modes=16, 
            width=64, 
            input_channels=4,  # u(x), x, t, eps
            depth = 3
            ):
        super().__init__()
        self.modes = modes
        self.width = width

        # Input layer
        self.linear_p = nn.Linear(input_channels, self.width)  # batchsize, N_gridpoints, input_channels [u(x), e, x]

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FNOBlock(self.modes, self.width) for _ in range(depth)  # Increase number of layers
        ])

        # Time embeddings
        self.time_layers = nn.ModuleList([
            TCBNEmbedding(self.width) for _ in range(depth)
        ])

        # Epsilon embeddings
        self.eps_layers = nn.ModuleList([
            TCBNEmbedding(self.width) for _ in range(depth)
        ])

        # Intermediate layers to manage skip connections
        self.skip_layers = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)
        ])

        # Final linear layers
        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = torch.nn.Tanh()

    def forward(self, x, eps, t):
        """
        Args:
            x: Initial condition (batch_size, x_size)
            eps: Epsilon values (batch_size, 1)
            t: Time points (batch_size, n_steps)
        Returns:
            Predictions at requested timepoints (batch_size, n_steps, x_size)
        """
        # Initial linear transformation
        x = self.linear_p(x)
        x = x.permute(0, 2, 1)  # to: batchsize, features, gridpoints

        # Process with Fourier layers, embeddings, and skip connections
        residual = x  # Store initial input for skip connection
        for i in range(len(self.fourier_layers)):
            x_space = self.activation(self.fourier_layers[i](x))
            x_time = self.time_layers[i](x, t)
            x_eps = self.eps_layers[i](x, eps)
            x_combined = x_space + x_time + x_eps

            # Add skip connection
            x = self.activation(self.skip_layers[i](x_combined)) + residual
            residual = x  # Update residual for the next layer

        x = x.permute(0, 2, 1)  # Back to: batchsize, gridpoints, features

        # Final linear layers
        x = self.linear_q(x)  # To: batchsize, gridpoints, features
        x = self.output_layer(x)

        return x


def get_loss_func():
    """
    TODO: Define custom loss function(s) for training
    Consider:
    - L2 loss on predictions
    - Physical constraints (energy, boundaries)
    - Gradient-based penalties
    """

    # add physics based loss with the follwoing code 

    # #Glinzbug- Landau free nergy

    grad_x = torch.gradient(u, dim=-1)
    energy = 0.5 * torch.mean(grad_x**2) + (1 / (4 * epsilon**2)) * torch.mean((u**2 - 1)**2)
    loss_energy = torch.mean(energy)

    # periodic bc loss

    boundary_loss = torch.mean((u[:, 0] - u[:, -1])**2)  # Example for periodic boundaries



    # L2_loss = nn.MSELoss()
    def residual_loss(u, epsilon): 
        """
        Residual=∂t∂u​−(Δu−ϵ^-2​(u^3−u))
        """

        NotImplementedError("Extract dx somehow and think about which axis t is in at this point")
        dx = x_grid[1] - x_grid[0]

        u_t = np.gradient(u, ..., axis = ...)
    
        # Compute Laplacian (Δu) with periodic boundary conditions
        laplacian = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        # Compute nonlinear term -(1/ε²)(u³ - u)
        non_linear_term = (1/epsilon**2) * (u**3 - u)

        residiual = u_t - (laplacian - non_linear_term)

        return torch.mean(residiual**2)    

    def smoothness_loss(u):
        """
        Smoothness Loss=∥∇u∥^2
        """

        NotImplementedError("Ensure gradient is taken along the correct dimension")

        u_x = torch.gradient(u, dim=-1)

        return torch.mean(u_x**2)


    return nn.MSELoss(), residual_loss, smoothness_loss

def get_optimizer(model, learning_rate):
    """
    TODO: Configure optimizer and learning rate schedule
    Consider:
    - Adam with appropriate learning rate
    - Learning rate schedule for curriculum
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

def train_step(model, batch, optimizer, loss_func):
    """
    TODO: Implement single training step
    1. Forward pass
    2. Loss computation
    3. Backward pass
    4. Optimizer step
    Return loss value
    """
    pass

def validation_step(model, batch, loss_func):
    """
    TODO: Implement single validation step
    Similar to train_step but without gradient updates
    Return loss value
    """
    pass

# Example usage:
if __name__ == "__main__":
    # Model initialization
    model = AllenCahnFNO(modes=16, width=64)
    
    # Sample data
    batch_size, x_size = 32, 128
    x = torch.randn(batch_size, x_size)
    eps = torch.randn(batch_size, 1)
    t = torch.linspace(0, 1, 4)[None].expand(batch_size, -1)
    
    # Forward pass
    output = model(x, eps, t)  # Should return (batch_size, n_steps, x_size)
