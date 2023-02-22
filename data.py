import pandas as pd
import sqlalchemy as db
from decouple import config
from nltk.tokenize import word_tokenize
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

    def get_local_data(self, path):
        self.df = pd.read_csv(path, sep=',',
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

    def process_data(self, x):
        x = x.lower()  # lowercase
        x = ' '.join(re.findall('(?<!\S)[a-z-]+(?=[,.!?:;]?(?!\S))', x))  # only keep words containing only letters

        return x

    def shape_data(self, file_name):
        self.df = self.df.sample(frac=1, random_state=42)  # shuffle data

        x = self.df.apply(lambda row: self.process_data(row['text']), axis=1)

        y = self.df['polarity']
        y = y.astype('int')

        np.savez(f'trainingandtestdata/{file_name}.npz', X_train=x, X_test=y)

        return x, y


if __name__ == '__main__':
    d = Data()
    d.get_local_data('trainingandtestdata/training.1600000.processed.noemoticon.csv')
    X_train, y_train = d.shape_data('train_data')
    d.get_local_data('trainingandtestdata/testdata.manual.2009.06.14.csv')
    X_test, y_test = d.shape_data('test_data')
