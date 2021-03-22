import pandas as pd
import os
import pyarrow


base = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/B3/'
files = os.listdir(base)

for file in files:
    df = pd.read_csv(os.path.join(base, file), delimiter = " ")
    df.columns = ['time', 'amplitude']
    print('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/{}.parquet'.format(file.split('.')[0]))

    df.to_parquet('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/{}.pq'.format(file.split('.')[0]), engine='pyarrow')