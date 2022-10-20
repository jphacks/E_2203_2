import pandas as pd
import numpy as np

path = 'jphack/test.csv'

df = pd.read_csv(path)


print(df)  #最初の5行のデータ表示
print(df.describe)  #データの概要を数値化

#大小関係を標準化する

from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
df_sc = sc.fit_transform(df.drop('food',axis = 1))
df_sc = pd.DataFrame(df_sc)
df.head()

print(df_sc)


from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

cos_similarity_matrix = pd.DataFrame(cosine_similarity(df_sc))

print(cos_similarity_matrix)


df_distance = cos_similarity_matrix[-1:]

i = len(cos_similarity_matrix) -1 #出力したラベルの行数を取得

print(i)#同じ数の列を消す

df_distance = df_distance.drop(df_distance.columns[i],axis = 1)


print(df_distance) #最後尾列の各点によるパラメータ、同じ数の列無し

a = df_distance.idxmax(axis = 1)
print(a)

a = a.iloc[-1] #seriesの値取り出し
print(a)

result_distance = df.at[df.index[a],'food']
print(result_distance)



df_distance1 = df_distance.drop(df_distance.columns[a],axis = 1)
print(df_distance1)

a = df_distance1.idxmax(axis = 1)
print(a)

a = a.iloc[-1] #seriesの値取り出し
print(a)

result_distance1 = df.at[df.index[a],'food']
print(result_distance1)



df_distance2 = df_distance1.drop(df_distance1.columns[a],axis = 1)
print(df_distance2)

a = df_distance2.idxmax(axis = 1)
print(a)

a = a.iloc[-1] #seriesの値取り出し
print(a)

result_distance2 = df.at[df.index[a],'food']
print(result_distance2)

print(result_distance) #結果(食べ物id)
print(result_distance1)
print(result_distance2)

#n = df.at[df.index[-1],'cluster'] #指定した番号のクラスタ番号取得

#cluster_n =  df[df['cluster']== n]