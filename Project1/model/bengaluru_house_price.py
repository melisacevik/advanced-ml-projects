import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load the data
df1 = pd.read_csv("/Users/melisacevik/PycharmProjects/ML-Advanced/model/bengaluru_house_prices.csv")
df1.head()

# Check the shape of the data
df1.shape

# Alan türü kategorilerinin her birindeki veri öğelerinin toplamı
#df1.groupby(gruplanacak_kolon)[işlem_yapılacak_kolon].işlem()

df1.groupby("area_type")["area_type"].agg("count")

# Modeli basit tutmak için kullanılmayacak kolonları çıkar

df2 = df1.drop(["area_type", "society", "balcony", "availability"], axis="columns")
df2.head()

# Data Cleaning: Handling NA values

df2.isnull().sum()

df3 = df2.dropna()
df3.isnull().sum()
df3.shape

df3["size"].unique()

df3["bhk"] = df3["size"].apply(lambda x: int(x.split(" ")[0]))

df3.head()

df3[df3.bhk > 20]

df3.total_sqft.unique()

# total_sqft kolonundaki verileri incelediğimizde bazı değerlerin aralık olarak girildiğini görebiliriz.
# Bu durumda bu aralıktaki değerlerin ortalamasını alarak tek bir değer olarak değiştireceğiz.
# '1133 - 1384' gibi değerlerin ortalamasını alarak 1258 gibi tek bir değer olarak değiştireceğiz.

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# float olmayan değerleri listele
df3[~df3["total_sqft"].apply(is_float)].head(10)

# total_sqft kolonundaki değerleri aralık olarak girilen değerlerin ortalamasını alarak tek bir değer olarak değiştirme

# 2 değeri input olarak alır ve aralarındaki değerlerin ortalamasını döndürür

def convert_sqft_to_num(x):
    tokens = x.split("-") # - işaretine göre ayırır
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

convert_sqft_to_num("2555 - 5555")

convert_sqft_to_num("34.46Sq. Meter") # Bu değer float olmadığı için None dönecek

df4 = df3.copy()
df4["total_sqft"] = df4["total_sqft"].apply(convert_sqft_to_num)
df4.head()

# 30. satırdaki total_sqft değeri 2475.0 olmalı
df4.loc[30]

# Feature Engineering

df5 = df4.copy()
# metrekare başına fiyatı hesapla : price lakh cinsinden olduğu için 100000 ile çarpıyoruz
df5["price_per_sqft"] = df5["price"]*100000 / df5["total_sqft"]
df5.head()

# Kaç konum var?
len(df4.location.unique())

# Konumları temizleme

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby("location")["location"].agg("count").sort_values(ascending=False)
location_stats

len(location_stats[location_stats <= 10])

location_stats_less_than_10 = location_stats[location_stats <= 10]

df5.location = df5.location.apply(lambda x: "other" if x in location_stats_less_than_10 else x)
len(df5.location.unique()) # 1304'ten 242'ye düştü

# Outlier Detection and Removal

# 1 yatak odası için outlier değerlerini tespit edeceğiz

# yatak odası sayısına göre tipik metrekare belirleme gibi bir threshold belirleyebiliriz
# 1 yatak odası için 300 metrekare

df5[df5.total_sqft / df5.bhk < 300].head()

df5.shape

df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
df6.shape

# price_per_sqft değerlerine bakarak outlier değerleri çıkaracağız

df6.price_per_sqft.describe()

# generic bir model kuracağımız için aşırı yüksek price_per_sqft değerlerini çıkaracağız.
# hatırlatma: ortalama, veri setinin merkezi eğilimini gösterir.
# standart sapma: verilerin ortalamadan ne kadar uzaklaştığını ölçen bir değerdir.
# SS düşükse, veriler ortalamaya yakın,
# SS yüksekse, veriler ortalamadan uzak (dağınık) demektir.
# ortalama += 1 standart sapma ve ortalama -= 1 standart sapma arasındaki değerleri alacağız
# bunun sebebi ortalamanın etrafında yoğunlaşan değerleri incelemek ve outlierı hariç tutmaktır.

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
                            # ortalama değerden standart sapma kadar yukarı ve aşağıda olan değerleri al
                            # ortalama - standart sapmadan büyükse ve ortalama + standart sapmadan küçük ve eşitse al
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)
df7.shape # 12502'den 10241'e düştü

# 2 ve 3 yatak odalı evlerin fiyatlarını karşılaştırarak outlierları tespit edeceğiz

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams["figure.figsize"] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color="blue", label="2 BHK", s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color="green", marker="+", label="3 BHK", s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    plt.show()

plot_scatter_chart(df7, "Hebbal")

# We should also remove properties where for same location,
# the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment
# (with same square ft area). What we will do is for a given location, we will build a
# dictionary of stats per bhk, i.e.
#
# {
#     '1' : {
#         'mean': 4000,
#         'std: 2000,
#         'count': 34
#     },
#     '2' : {
#         'mean': 4300,
#         'std: 2300,
#         'count': 22
#     },
# }

# bunu neden yapıyoruz? 2 yatak odalı evlerin fiyatı 3 yatak odalı evlerden daha fazla olmamalı
# eğer daha fazla ise outlier olarak kabul edeceğiz

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby("location"):
        bkh_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bkh_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bkh_stats.get(bhk-1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats["mean"])].index.values)

    return df.drop(exclude_indices, axis="index")

df8 = remove_bhk_outliers(df7)
df8.shape # 10241'den 7329'a düştü

plot_scatter_chart(df8, "Hebbal")

# kaç m2'lik dairemiz var?
# bunun için histogram çizdireceğiz

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()

# dağılım normal dağılıma yakın görünüyor

# Banyo sayısına göre outlierları tespit edeceğiz

df8.bath.unique()

df8[df8.bath > 10]

# 10'dan fazla banyosu olan evlerin outlier olduğunu düşünüyoruz

plt.hist(df8.bath, rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
plt.show()

# df9 = df8[df8.bath > df8.bhk + 2]
# banyo sayısının yatak odasının sayısından 2 fazla olduğu durumları outlier olarak kabul ediyoruz
# aradaki işaret > olduğu zaman
#            location       size  total_sqft  bath   price  bhk  price_per_sqft
# 1626  Chikkabanavar  4 Bedroom      2460.0   7.0    80.0    4     3252.032520
# 5238     Nagasandra  4 Bedroom      7000.0   8.0   450.0    4     6428.571429
# 6711    Thanisandra      3 BHK      1806.0   6.0   116.0    3     6423.034330
# 8411          other      6 BHK     11338.0   9.0  1000.0    6     8819.897689

df9 = df8[df8.bath < df8.bhk + 2]
df9.shape # 7329'dan 7251'e düştü

# gereksiz kolonları çıkar

df10 = df9.drop(["size", "price_per_sqft"], axis="columns")
df10.head()

# Model Building
# Kategorik verileri modelimizde kullanabilmek için One Hot Encoding yapacağız
# One Hot Encoding

dummies = pd.get_dummies(df10.location, dtype=int)
dummies.head()

# dummy tuzağından kaçınmak için bir kolonu çıkar. Biz de "other" kolonunu çıkaracağız. Sebebi ise "other" kolonu diğer tüm kolonların değeri 0 olduğunda "other" kolonunun değeri 1 olacak. Bu durumda diğer tüm kolonlar 0 olduğunda "other" kolonunun değeri 1 olacak ve bu durumda "other" kolonu diğer tüm kolonlarla aynı bilgiyi taşıyacak. Bu durumda "other" kolonunu çıkararak dummy tuzağından kaçınmış olacağız.
df11 = pd.concat([df10,dummies.drop("other",axis="columns")], axis="columns")
df11.head()

df12 = df11.drop("location", axis="columns")
df12.head(2)

df12.shape

X = df12.drop("price", axis="columns")
y = df12.price
y.head()

# model eğitimi için : train
# model performansı için : test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test) # 0.8452

# K-Fold Cross Validation
# Amaç: Modelin performansını ölçmek, modelin genelleme yeteneğini ölçmek, overfitting durumunu tespit etmek, modelin doğruluğunu artırmak

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv) # array([0.82430186, 0.77166234, 0.85089567, 0.80837764, 0.83653286])

# Diğer algoritmaları deneyerek modelin performansını artırmaya çalışacağız.
# GridSearchCV : en iyi parametreleri ve en iyi modeli bulmamıza yardımcı olur.

# en iyi modeli bulmak için fonksiyon oluşturuyoruz

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)

#                model  best_score                                        best_params
# 0  linear_regression    0.819001                           {'fit_intercept': False}
# 1              lasso    0.687429                {'alpha': 1, 'selection': 'cyclic'}
# 2      decision_tree    0.727752  {'criterion': 'friedman_mse', 'splitter': 'best'}

# en iyi modelin linear_regression olduğunu gördük

X.columns
np.where(X.columns=="2nd Phase Judicial Layout")[0][0]
# predict_price fonksiyonu oluşturuyoruz.

def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

predict_price("1st Phase JP Nagar",1000, 2, 2) # 83.865
predict_price("1st Phase JP Nagar",1000, 3, 3) # 86.80
predict_price("Indira Nagar",1000, 2, 2) # 181.278

# Flask sunucusunun ihtiyaç duyduğu dosyaları dışa aktaracağız

import pickle
with open("banglore_home_prices_model.pickle", "wb") as f:
    pickle.dump(lr_clf, f)

import json
columns = {
    "data_columns": [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))