import pandas as pd
import re
import unidecode

def clean_text(text):

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove mentions
    text = re.sub(r"@\S+", "", text)

    # remove hashtags
    text = re.sub(r"#\S+", "", text)

    # remove non-ascii characters
    text = unidecode.unidecode(text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)

    # remove extra spaces
    text = re.sub(r"\s+", ' ', text)

    # remove leading and trailing spaces
    text = text.strip()

    if len(text) <= 10:
        text = None

    return text

def main():

    data = pd.read_csv('unsupervised-pretraining/training.1600000.processed.noemoticon.csv', encoding='latin-1', index_col=False, header=None)
    data.drop(data.columns[[0, 1, 2, 3, 4]], axis=1, inplace=True)
    data[5] = data[5].apply(clean_text)
    data.dropna(inplace=True)

    data.to_csv('data/unsupervised-pretraining/training.txt', sep="\n", index=False, header=False)

if __name__ == "__main__":

    main()