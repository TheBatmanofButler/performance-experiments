import time
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

persisted_model_filename = "main.pth"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        
        return logits

def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    demo_basic(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()

    if local_rank == 0:
        check()

def demo_basic(local_world_size, local_rank):

    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 0 uses GPUs [0, 1, 2, 3] and
    # rank 1 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    )

    model = NeuralNetwork().cuda(device_ids[0])
    ddp_model = DDP(model, device_ids)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()

        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device_ids[0])
            y = y.to(device_ids[0])

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss = loss.item()
                current = (batch + 1) * len(X)

                if dist.get_rank() == 0:
                    print(f"loss: {loss:>7f} [{current:>5d}|{size:5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device_ids[0])
                y = y.to(device_ids[0])

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        if dist.get_rank() == 0:
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    total_duration = 0

    epochs = 100
    for t in range(epochs):
        start = time.time()
        
        if dist.get_rank() == 0:
            print(f"Epoch {t + 1}\n---------------------")
        
        train(train_dataloader, ddp_model, loss_fn, optimizer)
        test(test_dataloader, ddp_model, loss_fn)

        end = time.time()
        duration = round(end - start, 2)
        total_duration += duration

        if dist.get_rank() == 0:
            print(f"Epoch duration: {duration} seconds")

    if dist.get_rank() == 0:
        print(f"Done training! Total duration {round(total_duration, 2)} seconds")

    if dist.get_rank() == 0:
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in ddp_model.state_dict().items():
            # remove `module.`
            name = k[7:]
            new_state_dict[name] = v
        
        # load params
        model.load_state_dict(new_state_dict)

        torch.save(new_state_dict, persisted_model_filename)
        
        if dist.get_rank() == 0:
            print(f"Saved PyTorch model state to {persisted_model_filename}")

def check():
    print(f"Loading PyTorch model state from {persisted_model_filename}")

    device = torch.device('cuda:0')

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(persisted_model_filename, weights_only=True))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    spmd_main(args.local_world_size, args.local_rank)
