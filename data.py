import pandas as pd
import sqlalchemy as db
from decouple import config
from sklearn.model_selection import train_test_split
import re

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

    def get_database_data(self):
        ssl = {"ssl": {"ssl_ca": "ca.pem", "ssl_cert": "client-cert.pem", "ssl_key": "client-key.pem"}}
        password = config('MYSQL_OEGE_DATABASE_PASSWORD')
        engine = db.create_engine(f'mysql+pymysql://rijswij2:{password}@oege.ie.hva.nl/zrijswij2', connect_args=ssl)
        conn = engine.connect()

        self.df = pd.read_sql_table('tweet', con=conn)
        self.df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']

        conn.close()

    def shape_data(self):
        # x = self.df.loc[:, ['date', 'text']]
        # x['date'] = x['date'].apply(lambda x: re.search(r'\b\d{2}(?=:)', x).group(0))
        # x['date'] = x['date'].astype('int').values
        x = self.df['text'].values


        y = self.df['polarity']
        y = y.astype('int').values

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    d = Data()
    d.get_local_data()
    X_train, X_test, y_train, y_test = d.shape_data()
