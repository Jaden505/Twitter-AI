import pandas as pd
import sqlalchemy as db
import os

df = pd.read_csv('trainingandtestdata/training.1600000.processed.noemoticon.csv', sep=',', encoding='ISO-8859-1')
df.head()

# create db first in MySQL
ssl = {"ssl": {"ssl_ca": "ca.pem", "ssl_cert": "client-cert.pem","ssl_key": "client-key.pem"}}
password = os.environ.get('DATABASE_PASSWORD')
print(password)
engine = db.create_engine(f'mysql+pymysql://rijswij2:{password}@oege.ie.hva.nl/zrijswij2', connect_args=ssl)

df.to_sql(name='tweet', con=engine, if_exists='fail', index=False)
conn = engine.connect()

metadata = db.MetaData()
tweets = db.Table('tweet', metadata, autoload_with=engine)