import os
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from tqdm.notebook import tqdm

def visualize_images(images, mode, epoch):
    # Make a grid
    img_grid = torchvision.utils.make_grid(images.detach().cpu(), nrow=10)
    # Convert image from PyTorch tensor to numpy array
    os.makedirs(f"generated_images_{mode}", exist_ok = True)
    torchvision.utils.save_image(img_grid, os.path.join(f"generated_images_{mode}", f"epoch_{epoch}.png"))


def simplex_project(v, dim=5):
    pos_part = torch.nn.ReLU(inplace=True)
    radius=1.0
    mu, _ = torch.sort(v)
    cumul_sum = torch.divide(
        torch.flip(torch.cumsum(torch.flip(mu, [0]), dim=0), [0]) - radius, torch.arange(dim, 0, -1,device=mu.device))
    rho = torch.argmin(torch.where(mu > cumul_sum, torch.arange(dim,device=mu.device), dim))
    theta = cumul_sum[rho]
    v.add_(-theta)
    pos_part(v)

def run_alg(alg, f, x_init, y_init, n_iterations=1000, *args, **kwargs):
    x, y = x_init.clone().requires_grad_(True), y_init.clone().requires_grad_(True)
    x_sequence = [x.detach().numpy().copy()]
    y_sequence = [y.detach().numpy().copy()]
    for _ in range(n_iterations):
        alg(f, x, y, *args, **kwargs)
        x_sequence.append(x.detach().numpy().copy())
        y_sequence.append(y.detach().numpy().copy())
    return np.array(x_sequence), np.array(y_sequence)

def visualize_seq(L_x, L_y, dim_pair):
    dim_names = ["Rock", "Paper", "Scissors", "Lizard", "Spock"]
    plt.style.use('seaborn-v0_8-poster')
    plt.gca().add_patch(Polygon([(0, 0), (1.0, 0), (0, 1.0)],
                                     facecolor='y', alpha=0.1))
    plt.scatter(L_x[:, 0], L_x[:, 1], alpha=0.5, s=15)
    plt.scatter(L_y[:, 0], L_y[:, 1], alpha=0.5, s=15)
    path = L_x.T
    plt.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, width=0.0015, color='b', label="x")
    path = L_y.T
    plt.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, width=0.0015, color='r', label="y")
    plt.legend()

    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.xlabel(dim_names[dim_pair[0]])
    plt.ylabel(dim_names[dim_pair[1]])
    # plt.axis('equal')
    # plt.axis('off')
    plt.tight_layout()
    plt.show()



class GanTrainer():

    def __init__(self, batch_size, num_test_samples, data, noise, noise_dim, mode, device):
        self.data = data
        self.noise = noise
        self.noise_dim = noise_dim
        self.snapshots = []
        self.batch_size = batch_size
        self.num_test_samples = num_test_samples
        self.fixed_noise = self.noise.sample((num_test_samples, self.noise_dim, 1, 1)).to(device)
        self.step=0
        self.mode=mode
        self.device=device

    def _snapshot(self, g, epoch):
        """Save an image of the current generated samples"""
        with torch.no_grad():
            generated_images = g(self.fixed_noise)
            visualize_images(generated_images, self.mode, epoch)  


    def alternating(self, n_epochs, f, g, f_optim, g_optim, alternating_update, f_ratio):
        
        for epoch in tqdm(range(n_epochs), "Epoch"):
            bar = tqdm(self.data, "Batch", leave=False)
            for real, _ in bar:
                self.step += 1
                batch_size = real.size(0)
                real = real.to(self.device)
                noise = self.noise.sample((batch_size, self.noise_dim, 1, 1)).to(self.device)
                l= alternating_update(self.step, f, g, f_optim, g_optim, noise, real, d_ratio=f_ratio)
                bar.set_postfix_str(f"W1:{l:0.2} {'G' if self.step%f_ratio==0 else 'D'}")

            # g.eval()
            self._snapshot(g, epoch)


def train(f, g, f_optim, g_optim, alternating_update, batch_size=64, num_test_samples=100, 
          n_epochs=10, noise_dim=100, mode="spectral_norm", f_ratio=5, seed=1, device="cpu"):
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(device)

    # Load MNIST dataset
    transform = transforms.ToTensor()

    train_dataset = dsets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    #train_dataset = dsets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    #train_dataset = dsets.CelebA(root='./data', split = 'train', target_type = 'attr', transform=transform, download=True)

    indices = [i for i, (img, label) in enumerate(train_dataset) if label in  [0,2]]

    mnist_subset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = torch.utils.data.DataLoader(mnist_subset, batch_size=batch_size, shuffle=True)

    # plot the real data
    real_images, _ =  next(iter(torch.utils.data.DataLoader(mnist_subset, batch_size=num_test_samples, shuffle=True)))
    visualize_images(real_images, mode, -1)
    
    z = torch.distributions.Normal(0, 1)

    # Initialize trainer
    trainer = GanTrainer(batch_size,
                         num_test_samples=num_test_samples,
                         data=train_loader,
                         noise=z,
                         noise_dim=noise_dim, 
                         mode=mode, 
                         device=device)

    # train and save images
    trainer.alternating(n_epochs=n_epochs,
                        f=f,
                        g=g,
                        f_optim=f_optim,
                        g_optim=g_optim,
                        alternating_update=alternating_update,
                        f_ratio=f_ratio)
