import pandas as pd

# This file was used to convert the data from
# https://huggingface.co/datasets/SetFit/enron_spam]
# to create the more comfortable file we
# submitted called spam_data_pre_proccessed.csv


REGEX_RE = "re ?: ?"

REGEX_DATA_WHITESPACE = "((\n)?( ?- ?)+(\n)?)|(\n)"


def remove_whitespace(df):
    return df.replace(REGEX_DATA_WHITESPACE, ' ', regex=True).str.strip()


def remove_re(df):
    return df.replace(REGEX_RE, "", regex=True).str.strip()


def preproccess_data(filename):
    """
    :param filename:
    :return:
    """
    df = pd.read_csv(filename)
    df["Message"].fillna("", inplace=True)
    df = df.astype(
        {"Message": 'string', "Subject": 'string', 'Spam/Ham': 'string'})
    df["Message"] = remove_whitespace(df["Message"])
    df["Subject"] = remove_re(df["Subject"])
    df["Label"] = 0
    df.loc[df["Spam/Ham"] == "spam", "Label"] = 1
    df = df[["Subject", "Message", "Label"]]
    df.to_csv("spam_data_pre_proccessed.csv", index=False)


if __name__ == '__main__':
    preproccess_data("enron_spam_data.csv")
