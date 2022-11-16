import torch


def get_train_data(data_dir):
    train_tensor = torch.load(f"{data_dir}/train1989.pt")
    return train_tensor


def get_test_data(data_dir):
    test_tensor = torch.load(f"{data_dir}/test1989.pt")
    return test_tensor
