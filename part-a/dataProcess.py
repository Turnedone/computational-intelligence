import pandas as pd
import random
import numpy as np
import natsort
from natsort import natsorted

df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp']) # Διαβάζεται το αρχείο και αποθηκεύται σε μορφή pandas
df1 = df.copy() # Αντιγραφή σε νέο pandas dataframe για τη σύγκριση
gb = df.groupby('user_id') # Κατακερματισμός του dataframe σε groups ανά χρήστη
idgroups = [gb.get_group(x) for x in gb.groups] # Αποθήκευση των groups σε λίστα


for x in idgroups: #Για κάθε group/χρήστη
     avg = x['rating'].mean() #Υπολογισμός του μέσου όρου
     print(x['user_id'].values[0],avg) # Εκτύπωση των μέσων όρων πριν το centering
     df1.loc[df1.user_id == x['user_id'].values[0], 'rating'] -= avg #Centering


gb1 = df1.groupby('user_id') # Επαναλαμβάνεται η ίδια διαδικασία μέτα το centering
idgroups1 = [gb1.get_group(x) for x in gb1.groups]

for x in idgroups1:
    avg = x['rating'].mean()
    print(x['user_id'].values[0],f'{avg:.2f}')



print(df['rating'].std()) # Τυπική απόκλιση πριν
print(df1['rating'].std()) # και μέτα το centering
print(df1['rating'].max(),df1['rating'].min()) # Εύρος τιμών μετά το centering

df2 = df1.copy() # Δημιουργία αντίγραφου του dataframe
df2 = df2.pivot(index='user_id', columns='item_id', values=['rating']) # Τροποποίηση του πίνακα ώστε να υπάρχει αντιστιχία user_id-item_id με τιμές το rating


for i in range(1, len(df2.index) + 1): # Εμφωλευμένη for για τη διαχείρηση του κάθε κελιού του πίνακα
    tempUser = df2.loc[i, :]
    stdUser = tempUser.std(skipna=True) # Υπολογισμός τυπικής απόκλισης για κάθε χρήστη
    for j in range(1, len(df2.columns) + 1):
        dfcell = df2.loc[i, ('rating', j)]
        if pd.isnull(dfcell): # Έλεγχος για το αν υπάρχει rating για το συγκεκριμένο ζεύγος χρήστη-ταινίας
            newRating = random.uniform(-stdUser, stdUser) # Δημιουργία τυχαίας τιμής στο κατάλληλο διάστημα
            df2.loc[i, ('rating', j)] = newRating # Ανάθεση της τιμής

for i in range(1, len(df2.index) + 1): # Εμφωλευμένη for για τη διαχείρηση του κάθε κελιού του πίνακα
    tempUser = df2.loc[i, :]
    minUser = tempUser['rating'].min() # Εύρεση ελάχιστης τιμής rating για κάθε χρήστη
    maxUser = tempUser['rating'].max() # και μέγιστης
    for k in range(1, len(df2.columns) + 1):
        cell = df2.loc[i, ('rating', k)]
        df2.loc[i, ('rating', k)] = (cell - minUser) / (maxUser - minUser) # Κανονικοποίηση των τιμών στο διάστημα [0,1]

df2.to_csv("pivoted.csv")
df2 = pd.read_csv("pivoted.csv")

newHeader = df2.iloc[0] # Μορφοποίηση του πίνακα και απαλοιφή του multi-index που προέκυψε από το pivot
df2 = df2[1:]
df2.columns = newHeader
df2.rename(columns={'item_id' : 'user_id'}, inplace=True)

df2.reset_index(drop=True, inplace=True)

dummy = pd.get_dummies(df2['user_id'].astype(str)) # one-hot encoding του user_id
dummy = dummy.reindex(natsorted(dummy.columns), axis=1) # Ταξινόμιση βάση του user_id
df2 = pd.concat([df2, dummy], axis=1) # Προσθήκη των encoded user_id's ως columns στο dataframe
df2.drop('user_id', axis=1, inplace=True)

df2.to_csv("finalData.csv") # Εξαγωγή του τελικού dataframe για να γίνει το cross validation




