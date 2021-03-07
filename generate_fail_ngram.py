from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import pandas as pd
import csv

csv_data = [row for row in csv.reader(Path("./outputs/failed_sentences.tsv").open(), dialect='excel-tab')]
csv_data_0 = [row[0] for row in csv_data if row[1] == '0']
csv_data_1 = [row[0] for row in csv_data if row[1] == '1']

for i, c_data in enumerate([csv_data_0, csv_data_1]):
    c_vec = CountVectorizer(stop_words=[], ngram_range=(2, 3))
    # matrix of ngrams
    ngrams = c_vec.fit_transform(c_data)
    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_
    df_ngram = pd.DataFrame(
        sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
    ).rename(columns={0: "frequency", 1: "bigram/trigram"})

    df_ngram.to_csv(f'./outputs/fail_ngrams_{i}.csv')
