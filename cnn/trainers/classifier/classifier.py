# -----------------------------------------------------------------------------

from tensorboardX import SummaryWriter  # pip install tensorboardX
import torch
import os
import json
import numpy as np


def train(
    model: torch.nn.Module,
    train_data,
    test_data,
    output_dir,
    writer,
    optimizer,
    epochs=27,
    seed=1337,
    batch_size=128,
    loss_fn=torch.nn.MSELoss(),
    **kwargs,
):
    """Train the model on the training data"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)

    # if cuda is available, use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # print device
    print(f"Using device {device}")

    Xtr, Ytr = train_data

    for epoch in range(epochs):
        # perform one epoch of training
        model.train()
        for step_num in range(Xtr.size(0) // batch_size):

            # fetch a single example into a batch
            x = Xtr[step_num * batch_size : (step_num + 1) * batch_size]
            y = Ytr[step_num * batch_size : (step_num + 1) * batch_size]
            # x, y = Xtr[[step_num]], Ytr[[step_num]]

            # data is on the CPU, but the model is on the GPU
            x, y = x.to(device), y.to(device)

            # forward the model and the loss
            yhat = model(x)
            loss = loss_fn(yhat, y)

            # calculate the gradient and update the parameters
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # after epoch epoch evaluate the train and test error / metrics
        print(epoch + 1)
        eval(model, train_data, "train", writer, epoch)
        eval(model, test_data, "test ", writer, epoch)

    # save the model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))


def test(network, test_data, output_dir, writer, **kwargs):

    model_path = os.path.join(output_dir, "model.pt")
    assert os.path.exists(model_path), f"Model file {model_path} does not exist"

    # load the model
    model = torch.load(model_path)
    network.load_state_dict(state_dict=model)

    eval(network, test_data, "test ", writer, epoch=0)


def eval(model, data, split, writer, epoch):
    # eval the full train/test set, batched implementation for efficiency
    model.eval()
    X, Y = data

    # set same device as model
    device = next(model.parameters()).device
    X, Y = X.to(device), Y.to(device)

    Yhat = model(X)
    loss = torch.mean((Y - Yhat) ** 2)
    err = torch.mean((Y.argmax(dim=1) != Yhat.argmax(dim=1)).float())
    print(
        f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.size(0))}"
    )
    writer.add_scalar(f"error/{split}", err.item() * 100, epoch)
    writer.add_scalar(f"loss/{split}", loss.item(), epoch)
