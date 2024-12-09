# About the Dataset:
#
# 1. id: unique id for a news article
# 2. title: the title of a news article
# 3. author: author of the news article
# 4. text: the text of the article; could be incomplete
# 5. label: a label that marks whether the news article is real or fake:
#            1: Fake news
#            0: real News

# Importing the Dependencies
import numpy as np
import pandas as pd
import re
#bir belgedeki metni aramak için

from nltk.corpus import stopwords
#paragrafa değer katmayan sözcükler

from nltk.stem.porter import PorterStemmer
#Stemmer sözcüklerin kökünü oluşturmak için kullanılır.
# bir kelimeyi alır ve o kelimeden öncesini ve sonrasını kaldırır

from sklearn.feature_extraction.text import TfidfVectorizer
# metni özellik vektörlerine(sayılara) dönüştürmek için kullanılır.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', 1000) # max değişken
pd.set_option('display.width', 1000) # genişlik
pd.set_option('display.max_rows', 1000) # max gözlem

import nltk #nat lang. toolkit
nltk.download('stopwords')

print(stopwords.words("english"))

# Data Pre-processing

news_dataset = pd.read_csv("/Users/melisacevik/PycharmProjects/ML-Advanced/datasets/train.csv", sep=",")

news_dataset.shape #(20800, 5)

news_dataset.head()

news_dataset.isnull().sum()

# replacing the null values with empty string
news_dataset = news_dataset.fillna("")

# merging the author name and news title
news_dataset["content"] = news_dataset["author"]+" "+news_dataset["title"]

print(news_dataset["content"])

print(news_dataset)

# separating the data & label
X = news_dataset.drop(columns="label", axis=1)
Y = news_dataset["label"]

print(X)
print(Y)

# Stemming:
# Stemming is the process of reducing ad word to its Root word
# example:
# actor, actress, acting => act

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    # regular ex library'i içe aktardık ve arama yapmak için kullanışlı.
    # a-zA-Z ile sadece kelimeleri ve alfabeyi aldık. bu nedenle sayı ve noktalama iş. kaldırmış olduk
    # sayı ve noktlar yerine space
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    # ayırıp list haline getirme
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words("english")]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

news_dataset["content"] = news_dataset["content"].apply(stemming)

print(news_dataset["content"])

# separating the data and label

X = news_dataset["content"].values
Y = news_dataset["label"].values

print(X)
print(Y)
Y.shape

# X hala text. Pcnin anlayacağı hale getirmek için sayılara dönüştüreceğiz. (vektorizer)
# tf-idf vectorizer

# converting the textual data to numerical data

vectorizer = TfidfVectorizer()
vectorizer.fit(X) #x ile eşleştir

X = vectorizer.transform(X) #dönüştür

print(X)

# Splitting the dataset to training and test data
# X:bağımlı Y:bağımsız
# X_train ve y_train: Modelin öğrenmesi için kullanılır (%80).
# X_test ve y_test: Modelin doğruluğunu test etmek için kullanılır (%20).

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=2)

#Training the Model: Logistic Regression

# regressionda linear
# classification da logistic

model = LogisticRegression()

model.fit(X_train, Y_train)

# Evaluation
# accuracy score

#accuracy score on the training data
# Eğitim Setinde Doğruluk Hesaplama (Training Accuracy)
# bu durumda eğitim verisine özel doğruluğu ölçüyorsunuz,
# bu da modelin eğitim verisini ne kadar iyi öğrendiğini gösterir.
# binary classification'da logistic'i tercih et

X_train_prediction = model.predict(X_train) #bağımlı değişkeni tahmin et
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) #tahmin ettiğini doğrula

print("Accuracy score of the training data: ", training_data_accuracy)

#accuracy score on the test data
# Test Setinde Doğruluk Hesaplama (Test Accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy score of the test data: ", test_data_accuracy)

# Making a Predictive System
# modelin görmediği bir veri seçiyoruz

X_new = X_test[5]

prediction = model.predict(X_new) #modelin hiç görmeyip tahmin ettiği

print(prediction)

if (prediction[0] ==0):
    print("The news is Real")
else:
    print("The news is Fake")

print(Y_test[5]) #modelin hiç görmediği ve ayırdığımız sonuc

# bu ikisine bakarak doğru tahmin edip etmediğimizi test edebiliriz.