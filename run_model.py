import argparse
from dataset import create_tokens
import torch
from training import create_transform, get_model, SEQ_LENGTH
from utils import MODEL_LIST, PRIVACY_TYPE, ACCURACY_TYPE, device, NORMAL, \
    PRIVATE_TRAINING_BUDGET, PRIVATE_TRAINING, PRIVATE
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(description="run our models on emails to "
                                                 "check if they are spam or not")

    parser.add_argument('--subject', type=str, required=True,
                        help="subject for "
                             "mail")

    parser.add_argument('--msg', type=str, required=True,
                        help='content for mail')

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
                             'TRAINING - model is trained with DP-SGD but has '
                             'no budget')

    parser.add_argument("--accuracy_type", type=str, choices=ACCURACY_TYPE,
                        required=True, help='choose the accuracy type of model'
                                            'BEST - model will have higher '
                                            'accuracy but less privacy'
                                            'PRIVATE - model will have lower '
                                            'accuracy but will be more private.'
                                            ' Please notice that model with a '
                                            'privacy_training that is NORMAL '
                                            'cannot be PRIVATE.')

    args = parser.parse_args()
    model_type = args.model_type
    training = args.privacy_training
    accuracy_type = args.accuracy_type
    subject = args.subject
    msg = args.msg
    if training == NORMAL and accuracy_type == PRIVATE:
        raise Exception(
            "Model with a privacy_training that is NORMAL cannot be PRIVATE!")
    vocab = torch.load("vocab.pt")
    model, _ = get_model(vocab, model_type)
    model.eval()
    state_dict = torch.load(
        f"{accuracy_type}_{model_type}_{training}.pt".lower(),
        map_location=device)
    if training != NORMAL:
        # OPACUS changes the name for some reason
        state_dict = OrderedDict(
            [(k.replace('_module.', ""), v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict)
    transform = create_transform(SEQ_LENGTH, vocab)
    output = torch.sigmoid(model(transform(create_tokens(subject, msg))[None,
                                 :])).detach().cpu().item()
    print(f"the model output is:{output}")
    is_spam = round(output)
    if is_spam:
        print("this means that this mail is spam")
    else:
        print("this is not spam")


if __name__ == '__main__':
    main()
