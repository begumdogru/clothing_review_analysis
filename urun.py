# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 00:12:40 2021

@author: Begum Dogru
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv(r'D:\MASAÜSTÜ\yorumlar.csv')
veriler = veriler.dropna(how='any',axis=0)


a = veriler[['Clothing ID','Review Text']]

#yorumlardaki -varsa- emojileri, tırnak işaretlerini siliyoruz
a["Review Text"] = a["Review Text"].str.replace('[^\w\s]', '')

# sayılar
a["Review Text"] = a["Review Text"].str.replace('[\d]', '') 

#stopwords temizliyoruz
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
a["Review Text"] = a["Review Text"].astype(str)
a["Review Text"] = a["Review Text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#lemmi
from textblob import Word 
a["Review Text"] = a["Review Text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#buyuk-kucuk harf esitleme
a["Review Text"] = a["Review Text"].apply(lambda x: " ".join(x.lower() for x in x.split()))

#sentimental analiz
from textblob import TextBlob
a["sentiment"] = a["Review Text"].map(lambda x: TextBlob(x).sentiment.polarity)
for i in a["sentiment"]:
    if i < 0:
        a["sentiment"].replace(to_replace = i, value = "Negative", inplace= True)
    elif i > 0: 
        a["sentiment"].replace(to_replace = i, value = "Positive", inplace= True)
    else:
        a["sentiment"].replace(to_replace = i, value = "Notr", inplace= True)
        
#group by yapıyorum ürünleri kategorileştiriyorum
count =  a.groupby(["Clothing ID", "sentiment"])[["sentiment"]].count

plot_bar = a["sentiment"].value_counts().plot(kind= "bar")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.legend()
plt.title("Distribution of Sentiments")
plt.show()
print(count)

# yüzdesel olarak clothing id başına düşen yorum yüzdesi
yuzde = pd.crosstab(a["Clothing ID"],a["sentiment"]).apply(lambda r: (r/r.sum())*100, axis = 1)
print(yuzde)

plt.hist(veriler['Age'],bins=50,color='#008080',label="age") 
plt.xlabel("Age")
plt.ylabel("Distribution")
plt.legend()
plt.title("Distribution of Age")
plt.show()

df = pd.DataFrame(veriler)

aa = df.groupby('Department Name').count()['Clothing ID'].sort_values(ascending=False)
x= np.array(["Tops","Dresses","Bottoms","Intimate","Jackets","Trend"])
y= np.array([10468,6319,3799,1735,1032,119])
plt.title("The distribution of department")
plt.xlabel("Department name")
plt.ylabel("Count")
plt.bar(x,y)
plt.show()


b = df.groupby('Recommended IND').count()['Clothing ID'].sort_values(ascending=False)
x= np.array(["Yes","No"])
y= np.array([19314,4172])
plt.title("Distribution of Recommendations")
pie = plt.pie(y,autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.legend( loc = 'right', labels=x)
plt.show()


c = df.groupby('Class Name').count()['Department Name'].sort_values(ascending=False)
x= np.array(["Dress","Knts","Blous","Swts","Pnts","Jeans","FineGauge","Skrts","Jckts","Lounge","Swim","Outrwear","Shorts","Sleep","Legwear","Intimates","Layering","Trend","Casual Bottoms","Chemises"])
y= np.array([6319,4843,3097,1428,1388,1147,1100,945,704,691,350,328,317,228,165,154,146,119,2,1])
plt.title("The distribution of class")
plt.xlabel("class name")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.bar(x,y)
plt.show()


