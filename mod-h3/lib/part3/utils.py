import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.nn.modules.batchnorm import _BatchNorm

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.inner = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 10)  # Adjust output neurons for the number of classes in your dataset
        )

    def forward(self, x):
        return self.inner(x)


def evaluate(model, data_loader=None):
    torch.manual_seed(0)
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    
    if data_loader is None:
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        # test_dataset, _ = random_split(test_dataset, [0.2, 0.8], generator=torch.Generator().manual_seed(0))
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)
    
    #Evaluate on the validation set
    correct = 0
    total = 0
    model.eval()
    for img, label in data_loader:
        # img, label = img.to("cuda"), label.to("cuda")
        output = model(img)

        #Track accuracy
        correct += sum(torch.argmax(output, dim=-1) == label)
        total += label.shape[0]

    return float(correct/total)


def train_model(model, optimizer, optimize, max_epochs):
    torch.manual_seed(42)
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_dataset, _ = random_split(train_dataset, [0.2, 0.8], generator=torch.Generator().manual_seed(42))
    train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256)
    
    epoch = 0
    best_acc = 0
    while epoch < max_epochs:
        #Train on the training set
        model.train()
        for img, label in train_loader:
            # img, label = img.to("cuda"), label.to("cuda")
            optimize(model, optimizer, img, label)

        #Evaluate on the validation set
        acc = evaluate(model, val_loader)
        if acc > best_acc:
            best_acc = acc

        print(f"Epoch {epoch} with {round(acc, 3)} accuracy on the validation set.")
        epoch += 1

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
