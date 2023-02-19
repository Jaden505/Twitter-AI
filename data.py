import pandas as pd
import sqlalchemy as db
from decouple import config

class Data:
    def __init__(self):
        pass

    def create_database(self):
        self.get_local_data().to_sql(name='tweet', con=engine, if_exists='fail', index=False,
                                     dtype={
                                         'polarity': db.types.INTEGER(),
                                         'id': db.types.BIGINT(),
                                         'date': db.types.VARCHAR(255),
                                         'query': db.types.VARCHAR(255),
                                         'user': db.types.VARCHAR(900),
                                     })

    def get_local_data(self):
        df = pd.read_csv('trainingandtestdata/training.1600000.processed.noemoticon.csv', sep=',',
                         encoding='ISO-8859-1')
        df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
        return df

    def get_database_data(self):
        ssl = {"ssl": {"ssl_ca": "ca.pem", "ssl_cert": "client-cert.pem", "ssl_key": "client-key.pem"}}
        password = config('MYSQL_OEGE_DATABASE_PASSWORD')
        engine = db.create_engine(f'mysql+pymysql://rijswij2:{password}@oege.ie.hva.nl/zrijswij2', connect_args=ssl)
        conn = engine.connect()
        metadata = db.MetaData()

        tweets = db.Table('tweet', metadata, autoload_with=engine)

        df = pd.read_sql_table('tweet', con=conn)
        df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']

        conn.close()
        print(df.head(2))

        return df


if __name__ == '__main__':
    d = Data()
    d.get_database_data()

