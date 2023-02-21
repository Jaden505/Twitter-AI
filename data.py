import pandas as pd
import sqlalchemy as db
from decouple import config
from sklearn.model_selection import train_test_split
import re
import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords


class Data:
    def __init__(self):
        self.df = None

    def create_database(self, engine):
        self.df.to_sql(name='tweet', con=engine, if_exists='fail', index=False,
                                     dtype={
                                         'polarity': db.types.INTEGER(),
                                         'id': db.types.BIGINT(),
                                         'date': db.types.VARCHAR(255),
                                         'query': db.types.VARCHAR(255),
                                         'user': db.types.VARCHAR(900),
                                     })

    def get_local_data(self):
        self.df = pd.read_csv('trainingandtestdata/training.1600000.processed.noemoticon.csv', sep=',',
                         encoding='ISO-8859-1')
        self.df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']

        return self.df

    def get_database_data(self):
        ssl = {"ssl": {"ssl_ca": "ca.pem", "ssl_cert": "client-cert.pem", "ssl_key": "client-key.pem"}}
        password = config('MYSQL_OEGE_DATABASE_PASSWORD')
        engine = db.create_engine(f'mysql+pymysql://rijswij2:{password}@oege.ie.hva.nl/zrijswij2', connect_args=ssl)
        conn = engine.connect()

        self.df = pd.read_sql_table('tweet', con=conn)
        self.df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']

        conn.close()

        return self.df

    def get_stopwords(self):
        nltk_words = set(stopwords.words('english'))
        gensim_words = set(STOPWORDS)
        sklearn_words = set(ENGLISH_STOP_WORDS)
        return list(nltk_words | gensim_words | sklearn_words)

    def process_data(self, x, pattern, digits_urls_mentions):
        x = x.lower()
        x = re.sub(r'\b(?:{})\b|{}|\s+'.format(self.get_stopwords(), r'[^\w\s]'), ' ', x).strip()
        x = pattern.sub(' ', x)
        x = digits_urls_mentions.sub('', x)

        return x

    def shape_data(self):
        print("Unprocessed data: ", self.df['text'].values[0], "\n", self.df['text'].values[1], "\n", self.df['text'].values[2], "\n", self.df['text'].values[3], "\n", self.df['text'].values[4], "\n", self.df['text'].values[5], "\n", self.df['text'].values[6], "\n", self.df['text'].values[7], "\n", self.df['text'].values[8], "\n", self.df['text'].values[9], "\n", self.df['text'].values[10], "\n", self.df['text'].values[11], "\n", self.df['text'].values[12], "\n", self.df['text'].values[13], "\n", self.df['text'].values[14], "\n")

        pattern = re.compile(r'[^\w\s]')
        digits_urls_mentions = re.compile(r'\b\d+\b|\w*\d\w*|\bhttps?:\/\/\S+|\B@\w+')
        x = self.df['text'].apply(lambda text: self.process_data(text, pattern, digits_urls_mentions)).values

        print("Processed data: ", x[0], "\n", x[1], "\n", x[2], "\n", x[3], "\n", x[4], "\n", x[5], "\n", x[6], "\n", x[7], "\n", x[8], "\n", x[9], "\n", x[10], "\n", x[11], "\n", x[12], "\n", x[13], "\n", x[14], "\n")

        y = self.df['polarity']
        y = y.astype('int').values

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        np.savez('trainingandtestdata/shaped_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    d = Data()
    d.get_local_data()
    X_train, X_test, y_train, y_test = d.shape_data()
