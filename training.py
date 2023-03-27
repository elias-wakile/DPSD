from utils import *
from dataset import EmailDataset, get_datasets_split_and_vocab, PD_TOKEN
from torchtext.transforms import ToTensor, VocabTransform, Truncate, \
    PadTransform
from torchvision.transforms import Compose, Lambda
from models import AttentionClassifier, LSTMModel
import opacus
import opacus.optimizers.optimizer
import argparse


def create_transform(length, vocab):
    """
    create a transform object to convert list of tokens into tensor
    :param length: length of each row
    :param vocab: vocabulary
    :return: transform to convert
    """
    transform_list = [VocabTransform(vocab), Truncate(length),
                      ToTensor(vocab[PD_TOKEN], dtype=torch.long),
                      PadTransform(length, vocab[PD_TOKEN]),
                      Lambda(lambda x: x.to(device))]
    transform = Compose(transform_list)
    return transform


def create_dataset_object(length=None):
    """
    creates an EmailDataset object for train,val,test
    :param length: length of each row
    :return: train,val,test,vocab
    """
    train, val, test, vocab = get_datasets_split_and_vocab()
    transform = create_transform(length, vocab)
    train_ds = EmailDataset(train[0], train[1], transform)
    val_ds = EmailDataset(val[0], val[1], transform)
    test_ds = EmailDataset(test[0], test[1], transform)
    return train_ds, val_ds, test_ds, vocab


def get_model(vocab, model_type):
    model = None
    lr = None
    if model_type == "LSTM":
        model = LSTMModel(len(vocab),
                          embedding_size=128,
                          hidden_layer=64,
                          num_layers=2,
                          output_layer=1,
                          dropout=0.04906
                          ).to(device)
        lr = 0.008576

    elif model_type == "LSTM_ATTN":
        model = AttentionClassifier(len(vocab),
                                    embedding_size=128,
                                    hidden_size=64,
                                    num_layers=2,
                                    dropout=0.04906,
                                    num_of_heads=4,
                                    seq_length=SEQ_LENGTH).to(device)
        lr = 0.008576
    return model, lr


def train(model_type, private_mode=None, max_grad_norm=None, delta=None,
          epsilon=None, noise_multiplier=None):
    train_ds, val_ds, test_ds, vocab = create_dataset_object(SEQ_LENGTH)
    train_dt = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_dt = DataLoader(val_ds, BATCH_SIZE)
    test_dt = DataLoader(test_ds, BATCH_SIZE)
    criterion = torch.nn.BCEWithLogitsLoss()
    model, lr = get_model(vocab, model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    privacy_engine = None
    if private_mode == "BUDGET":
        privacy_engine = opacus.PrivacyEngine()
        model, optimizer, train_dt = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dt,
            max_grad_norm=max_grad_norm,
            target_delta=delta,
            target_epsilon=epsilon,
            epochs=10
        )
    elif private_mode == "TRAINING":
        privacy_engine = opacus.PrivacyEngine()
        model, optimizer, train_dt = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dt,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier
        )
    for e in range(10):
        avg_batch_loss = 0
        model.train()
        for x, y in train_dt:
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(), y.float())
            avg_batch_loss += loss.detach().cpu().item()
            loss.backward()
            optimizer.step()

        avg_batch_loss = avg_batch_loss / len(train_dt)
        train_acc, val_acc = get_metrics(model, train_dt, val_dt)
        print_step(e, avg_batch_loss, train_acc, val_acc)

    test_accuracy = get_accuracy(model, test_dt)
    print(f"test accuracy:{test_accuracy}")
    if delta and private_mode != NORMAL:
        print(f"delta:{delta},epsilon{privacy_engine.get_epsilon(delta)}")


def main():
    parser = argparse.ArgumentParser(description="train models and get their "
                                                 "values")
    parser.add_argument('--model_type', type=str, choices=MODEL_LIST,
                        required=True, help='choose a model to run'
                                            'LSTM - LSTM model'
                                            'LSTM_ATTN - LSTM model with '
                                            'attention')

    parser.add_argument('--privacy_training', type=str, choices=PRIVACY_TYPE,
                        required=True,
                        help='choose the privacy mode of the model'
                             'NORMAL - no privacy mode'
                             'BUDGET - model has a budget of (epsilon,delta) '
                             'and trained to match it'
                             'Training - model is trained with DP-SGD but has '
                             'no budget')

    parser.add_argument("--max_grad_norm", type=float, default=1
                        , help="maximum gradient value used for DP-SGD")

    parser.add_argument("--delta", type=float, default=1e-4
                        , help="delta for (epsilon,delta)-DP")

    parser.add_argument("--eps", type=float, default=1,
                        help="epsilon for (epsilon,delta)-DP used only in " \
                             "BUDGET mode ")
    parser.add_argument("--noise_multiplier", type=float, default=0.5,
                        help="random noise for DP-SGD used only in TRAINING "
                             "mode")

    args = parser.parse_args()
    model_type = args.model_type
    training = args.privacy_training
    max_grad_norm = args.max_grad_norm
    delta = args.delta
    epsilon = args.eps
    noise_multiplier = args.noise_multiplier

    train(model_type, training, max_grad_norm, delta, epsilon, noise_multiplier)


if __name__ == '__main__':
    main()
