import pandas as pd
import natsort
import random
import numpy as np
from natsort import natsorted

df = pd.read_csv('encoded.csv')
df.drop(df.index[0],inplace=True)
df1 = df.iloc[:, 1:1683]
df2 = df.iloc[:,1683:]
print(df1)
print(df2)
print(df2.iloc[:,:].values)
print(df1.iloc[:,:].values)


# df1 = df.sample(n=100)
# header = ["user_id", "item_id", "rating"]
# df1.to_csv('testData.csv', columns=header, index=False)
