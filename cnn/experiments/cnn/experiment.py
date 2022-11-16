from ...models.covnet import Net
from ...data.mnist_1989 import get_train_data, get_test_data
from ...trainers.classifier import train, test
from tensorboardX import SummaryWriter
import os
import torch
import sys


network = Net()

optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

save_dir: str = os.environ.get("SAVE_DIR")

data_dir: str = os.environ.get("DATA_DIR")

writer = SummaryWriter(save_dir)

command = sys.argv[1]

if command == "train":
    train_data = get_train_data(data_dir)
    test_data = get_test_data(data_dir)
    train(network, train_data, test_data, save_dir, optimizer=optimizer, writer=writer)
elif command == "test":
    test_data = get_test_data(data_dir)
    test(network, test_data, save_dir, writer=writer)
