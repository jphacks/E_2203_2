import numbers
import pandas as pd
import numpy as np
from pathlib import Path


#データフレーム読み込み

path = 'jphack/test.csv'

df = pd.read_csv(path)

print(df)  #最初の5行のデータ表示


#大小関係を標準化する

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
df_sc = sc.fit_transform(df.drop('food',axis = 1))
df_sc = pd.DataFrame(df_sc)
df.head()



#クラスタリング実行

from sklearn.cluster import KMeans

model = KMeans(n_clusters = 3, random_state =1)

model.fit(df_sc)



cluster = model.labels_

print(df)


df['cluster'] = cluster #dfに勝手にclusterが付け加えられてる…

df.to_csv("kmeans_model.csv") #xgboostの学習用にcsvファイル保存

print(df)

print(cluster)

print(df)

#分布グラフ表示

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca = PCA(n_components=2, random_state=1) #主成分分析の次元削減(PCA)実行
pca.fit(df_sc)
feature = pca.transform(df_sc)

import matplotlib.pyplot as plt #グラフ作成

plt.figure(figsize=(6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=cluster)
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()


#一番下のデータと同じクラスタのデータを取得

print(df.tail(1))  #tail(1):末尾の行を指定

print(df[-1:]) #これでも末尾の行指定ができる

print(df.at[df.index[-1],'cluster']) #末尾の行のクラスタを出力するやつ

n = df.at[df.index[-1],'cluster'] #指定した番号のクラスタ番号取得

cluster_n =  df[df['cluster']== n]

print(cluster_n)

print(df)



#同じクラスタのアイテムに絞ってXGBoost

# XGBoost

print(df)

from sklearn.model_selection import train_test_split
import xgboost as xgb

df_x = df.drop(['cluster'], axis=1)
df_y = df['cluster']

print(df_x)
print(df_y)

train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, random_state=1)

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest  = xgb.DMatrix(test_x, label=test_y)

params = {'objective':'reg:linear',
          'silent':1,
          'random_state':1}
num_round = 50

watchlist = [(dtrain, 'train'), (dtest, 'test')]

#学習
model = xgb.train(params, dtrain, num_round, verbose_eval=10, evals=watchlist)



#推論実行,特徴から推論したラベルをぶち込む

#new_dataのdfの部分は、新しく生成されたcsvファイルをデータフレームにしてから挿入

print(df)

new_data = df[-1:].drop(['cluster'], axis=1)

print(new_data)

pred = model.predict(xgb.DMatrix(new_data))

print(pred)


new_label = pred[0]

new_label = round(new_label)


print(new_label)


#モデルの保存

import os
import pickle

file_name = "xgb_model.pickle"

pickle.dump(model, open(file_name, "wb"))



#モデルのサイズ確認
print(round(os.path.getsize(file_name)/1024/1024,1),'Mb')



#モデル読み込み 
loaded_model = pickle.load(open(file_name,'rb'))

pred = loaded_model.predict(xgb.DMatrix(new_data)) #newdataに,送られてきたデータ(dfにして)を挿入
print(pred)

#配列から数値だけ取り出す
new_label = pred[0]

new_label = round(new_label)

#最終的に送信するデータ
recommend_data = df[df['cluster']== new_label]

print(recommend_data)

i = len(recommend_data) #出力したラベルの行数を取得

print(len(recommend_data))


recommend_df = recommend_data['food'].value_counts()

recommend_df.rename()

print(recommend_df)




