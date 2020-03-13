# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:24:21 2020

@author: Matthew Rosencrans
"""
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv (r'C:\Users\Matthew Rosencrans\Desktop\TSS_Calculator\Set1.csv')

isY = lambda x:int(x=='Y')
countCountTot = lambda row: (row['CE']) + (row['CP1']) + (row['CP2']) + (row['CP3'])

df['CountTot'] = df.apply(countCountTot,axis=1)

df1 = df.loc[(df.CountTot >= int(1))]
train1, test1 = train_test_split(df1, test_size=0.2)
df1strong = train1.loc[(df.binding_affinity >= int(20))]

df2 = df.loc[(df.CountTot >= int(2))]
train2, test2 = train_test_split(df2, test_size=0.2)
df2strong = train2.loc[(df.binding_affinity >= int(20))]

df3 = df.loc[(df.CountTot >= int(3))]
train3, test3 = train_test_split(df3, test_size=0.2)
df3strong = train3.loc[(df.binding_affinity >= int(20))]

df4 = df.loc[(df.CountTot >= int(4))]
train4, test4 = train_test_split(df4, test_size=0.2)
df4strong = train4.loc[(df.binding_affinity >= int(20))]

df5 = df.loc[(df.CountTot >= int(5))]
train5, test5 = train_test_split(df5, test_size=0.2)
df5strong = train5.loc[(df.binding_affinity >= int(20))]

df6 = df.loc[(df.CountTot >= int(6))]
train6, test6 = train_test_split(df6, test_size=0.2)
df6strong = train6.loc[(df.binding_affinity >= int(20))]

df7 = df.loc[(df.CountTot >= int(7))]
train7, test7 = train_test_split(df7, test_size=0.2)
df7strong = train7.loc[(df.binding_affinity >= int(20))]

df8 = df.loc[(df.CountTot >= int(8))]
train8, test8 = train_test_split(df8, test_size=0.2)
df8strong = train8.loc[(df.binding_affinity >= int(20))]

df9 = df.loc[(df.CountTot >= int(9))]
train9, test9 = train_test_split(df9, test_size=0.2)
df9strong = train9.loc[(df.binding_affinity >= int(20))]

df10 = df.loc[(df.CountTot >= int(10))]
train10, test10 = train_test_split(df10, test_size=0.2)
df10strong = train10.loc[(df.binding_affinity >= int(20))]

train1.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD1\train1.csv', index=False)
train2.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD2\train2.csv', index=False)
train3.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD3\train3.csv', index=False)
train4.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD4\train4.csv', index=False)
train5.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD5\train5.csv', index=False)
train6.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD6\train6.csv', index=False)
train7.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD7\train7.csv', index=False)
train8.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD8\train8.csv', index=False)
train9.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD9\train9.csv', index=False)
train10.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD10\train10.csv', index=False)
test1.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD1\test1.csv', index=False)
test2.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD2\test2.csv', index=False)
test3.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD3\test3.csv', index=False)
test4.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD4\test4.csv', index=False)
test5.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD5\test5.csv', index=False)
test6.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD6\test6.csv', index=False)
test7.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD7\test7.csv', index=False)
test8.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD8\test8.csv', index=False)
test9.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD9\test9.csv', index=False)
test10.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD10\test10.csv', index=False)
df1strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD1\df1strong.csv', index=False)
df2strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD2\df2strong.csv', index=False)
df3strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD3\df3strong.csv', index=False)
df4strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD4\df4strong.csv', index=False)
df5strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD5\df5strong.csv', index=False)
df6strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD6\df6strong.csv', index=False)
df7strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD7\df7strong.csv', index=False)
df8strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD8\df8strong.csv', index=False)
df9strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD9\df9strong.csv', index=False)
df10strong.to_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD10\df10strong.csv', index=False)



#from main import get_linreg_models
