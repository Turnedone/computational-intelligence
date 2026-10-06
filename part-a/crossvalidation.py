import pandas as pd
import random
import numpy as np

df = pd.read_csv('finalData.csv')

df.drop(df.index[0],inplace=True)
shuffled = df.sample(frac=1)
splitData = np.array_split(shuffled, 5) # Τυχαία ανακατανομή του πίνακα και διαμέριση σε 5 τμήματα

firstFlag = 0


for i in range(len(splitData)): # Εμφωλευμένη for για τη δημιουργία κάθε fold
    firstFlag = 1
    splitData[i].to_csv("./traintest/test" + str(i) + ".csv", index=False) # To test μέρος του fold
    for j in range(len(splitData)):
        if j != i:
            if firstFlag == 1:
                splitData[j].to_csv("./traintest/train" + str(i) + ".csv", index=False) # To train μέρος του fold
                firstFlag = 0
            else:
                splitData[j].to_csv("./traintest/train" + str(i) + ".csv", mode='a', header=False, index=False)


