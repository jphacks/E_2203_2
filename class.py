import numbers
import pandas as pd
#データフレーム読み込み
df = pd.read_csv("jphack/test.csv")

print(df)  #最初の5行のデータ表示
print(df.describe)  #データの概要を数値化

#でかいデータの大小関係を標準化する

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_sc = sc.fit_transform(df)
df_sc = pd.DataFrame(df_sc, columns=df.columns)
df.head()

columns = ['id','food','time','temperature','humidity']
df = df[columns]

df.isnull().sum()


#クラスタリング実行

from sklearn.cluster import KMeans

model = KMeans(n_clusters = 3, random_state =1)

model.fit(df_sc)

cluster = model.labels_


df['cluster'] = cluster

print(cluster)

#分布グラフ表示

from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=1)
pca.fit(df_sc)
feature = pca.transform(df_sc)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=cluster)
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()

#一番下のidから属するクラスを選択

print(df.tail(1))  #tail(1):末尾の行を指定

print(df[-1:]) #これでも末尾の行指定ができる

print(df.at[df.index[-1],'cluster'])

n = df.at[df.index[-1],'cluster']

df[df['cluster']== n]

print(df[df.cluster == n])



