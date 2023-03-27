import torch
from torch.utils.data import DataLoader

device = "cpu"
BATCH_SIZE = 64


SEQ_LENGTH = 128

LSTM = "LSTM"
LSTM_ATTN = "LSTM_ATTN"

MODEL_LIST = [LSTM, LSTM_ATTN]

NORMAL = "NORMAL"
PRIVATE_TRAINING_BUDGET = "BUDGET"
PRIVATE_TRAINING = "TRAINING"
PRIVACY_TYPE = [NORMAL, PRIVATE_TRAINING, PRIVATE_TRAINING_BUDGET]


BEST = "BEST"
PRIVATE = "PRIVATE"

ACCURACY_TYPE = [BEST,PRIVATE]


def get_metrics(model, train_dt, val_dt):
    model.eval()
    train_acc = get_accuracy(model, train_dt)
    val_acc = get_accuracy(model, val_dt)
    return train_acc, val_acc


def get_accuracy(model, dt: DataLoader):
    with torch.no_grad():
        avg_acc = 0
        for x, y in dt:
            y = y.float().to(device)
            output = model(x)
            avg_acc += torch.sum(
                torch.eq(torch.round(torch.sigmoid(output)).squeeze(),
                         y)).item()
    bs = dt.batch_size if dt.batch_size is not None else BATCH_SIZE
    return avg_acc / (len(dt) * bs)


def print_step(epoch, loss, train_acc, val_acc):
    print(f"epoch:{epoch + 1}")
    print(f"loss:{loss}")
    print(f"train_acc:{train_acc}")
    print(f"val_acc:{val_acc}")
    print()
