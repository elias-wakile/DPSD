import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re

SUB_TOKEN = "<sub>"
EOSSUB_TOKEN = "<eossub>"
EOS_TOKEN = "<eos>"
PD_TOKEN = "<pd>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [SUB_TOKEN, EOSSUB_TOKEN, EOS_TOKEN, PD_TOKEN, UNK_TOKEN]

REGEX_RE = "re ?: ?"

REGEX_DATA_WHITESPACE = "((\n)?( ?- ?)+(\n)?)|(\n)"

NOT_SPECIAL_CHARS = "[^a-zA-Z0-9.?! @]"

HYPER_LINK_REGEX = r"https?:?\?S+"

WHITE_SPACE_REGEX = re.compile(r"\s*([^a-zA-Z0-9])\s*")

TOKENIZER = get_tokenizer("basic_english")


def remove_special_chars_whitespace(msg: str) -> str:
    """
    removes leading and trailing white space from special chars
    :param msg: msg
    :return: msg witheout leading and trailing whitespace
    """
    return WHITE_SPACE_REGEX.sub(r"\1", msg)


def remove_reply_whitespace(msg: str) -> str:
    """
    some msg have \n--------\n when you reply to a msg. this function removes
    this from the msg
    :param msg: msg
    :return: msg without \n-----\n
    """
    return re.sub(REGEX_DATA_WHITESPACE, ' ', msg).strip()


def remove_re(subject: str) -> str:
    """
    remove "re:" at start of subject
    :param subject: subject
    :return: subject without re
    """
    return re.sub(REGEX_RE, ' ', subject).strip()


def remove_hyperlink(msg: str):
    """
    removes hyperlinks in msg
    :param msg: msg
    :return: msg without hyperlinks
    """
    return re.sub(HYPER_LINK_REGEX, "", msg)



def remove_special(text: str):
    """
    remove speical chars in text
    :param text: remove special chars in text
    :return: text without special chars
    """
    return re.sub(NOT_SPECIAL_CHARS, " ", text)


def create_full_email_tokens(subject, message, tokenizer):
    """
    creates a single list of string tokens
    :param subject: subject
    :param message: msg
    :param tokenizer: tokenizer
    :return: list of tokens representing the email
    """
    return [SUB_TOKEN] + tokenizer(subject) + [EOSSUB_TOKEN] + tokenizer(
        message) + [EOS_TOKEN]


def create_tokens(subject, msg):
    """
    does the whole preprocessing for a single mail
    :param subject: subject
    :param msg: msg
    :return: list of preprocessed tokens
    """
    msg = msg.lower()
    msg = remove_reply_whitespace(msg)
    msg = remove_special_chars_whitespace(msg)
    msg = remove_hyperlink(msg)
    msg = remove_special(msg)
    subject = remove_re(subject.lower())
    return create_full_email_tokens(subject, msg, tokenizer=TOKENIZER)


def create_email_and_labels(tokenizer):
    """
    does the whole preprocessing for the the whole dataframe
    :param tokenizer: tokenizer to use
    :return: email array(list of list containing tokens), labels
    """
    df = pd.read_csv("spam_data_pre_proccessed.csv", keep_default_na=False,
                     dtype={
                         "Subject": 'string',
                         "Message": 'string',
                         "label": 'int32'
                     })
    df["Subject"] = df["Subject"].str.lower()
    df["Message"] = df["Message"].str.lower()
    df["Message"] = df["Message"].apply(remove_special_chars_whitespace)
    df["Message"] = df["Message"].replace(HYPER_LINK_REGEX, "", regex=True)
    df["Message"] = df["Message"].replace(NOT_SPECIAL_CHARS, "", regex=True)

    df["emails"] = df.apply(lambda x: create_full_email_tokens(x["Subject"],
                                                               x["Message"],
                                                               tokenizer),
                            axis=1)
    return df["emails"].values, df["Label"].values


class EmailDataset(Dataset):
    """
    class used to represent the email dataset
    """
    def __init__(self, emails, labels, transform=None):
        self.emails = emails
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.emails)

    def __getitem__(self, idx):
        item = self.emails[idx]
        if self.transform:
            item = self.transform(item)
        return item, self.labels[idx]


def get_datasets_split_and_vocab():
    """
    a function that returns a (train,val,test) split and data
    :return: train,val,test,vocabulary
    """
    emails, labels = create_email_and_labels(TOKENIZER)
    train_emails, val_emails, test_emails = random_split(emails,
                                                         [0.8, 0.1, 0.1],
                                                         generator=torch.Generator().manual_seed(
                                                             42))
    train_labels, val_labels, test_labels = labels[train_emails.indices].copy(), \
                                            labels[val_emails.indices].copy(), \
                                            labels[test_emails.indices].copy()
    vocab = build_vocab_from_iterator(train_emails, min_freq=50,
                                      specials=SPECIAL_TOKENS,
                                      special_first=True)
    vocab.set_default_index(vocab[UNK_TOKEN])

    train = train_emails, train_labels
    validation = val_emails, val_labels
    test = val_emails, val_labels
    return train, validation, test, vocab
