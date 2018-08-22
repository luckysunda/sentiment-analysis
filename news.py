
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import sklearn
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse


# In[2]:


news_train = pd.read_csv('data/news-train.csv')
news_train = news_train[['title','description','label']]


# In[3]:


news_test = pd.read_csv('data/news-train.csv')
news_test = news_test[['title','description']]


# In[4]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


# In[6]:


#preprocessing Train data
import re
ps = PorterStemmer()
stops = set(stopwords.words("english"))

clean_train_title=[]
clean_train_description=[]

for review in news_train["title"]:
    review = re.sub("[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    #words = [ps.stem(w) for w in words ]
    clean_train_title.append(" ".join(words))

    

for review in news_train["description"]:
    review = re.sub("[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    #words = [ps.stem(w) for w in words ]
    clean_train_description.append(" ".join(words))


# In[7]:


#preprocessing test data

clean_test_title=[]
clean_test_description=[]

for review in news_test["title"]:
    review = re.sub("[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    words = [ps.stem(w) for w in words ]
    clean_test_title.append(" ".join(words))

    

for review in news_test["description"]:
    review = re.sub("[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    words = [ps.stem(w) for w in words ]
    clean_test_description.append(" ".join(words))


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
count_vec1 = TfidfVectorizer(stop_words='english')
count_vec2 = TfidfVectorizer(stop_words='english')
train_count2=count_vec2.fit_transform(clean_train_description)
train_count1=count_vec1.fit_transform(clean_train_title)
train_count=sparse.hstack((train_count1,train_count2))
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
from sklearn.cross_validation import KFold,cross_val_score
scores=cross_val_score(clf,train_count,news_train["label"],cv=10,scoring='accuracy')


# In[9]:


scores.mean()


# In[10]:



test_count2=count_vec2.transform(clean_test_description)
test_count1=count_vec1.transform(news_test.title)
test_count=sparse.hstack((test_count1,test_count2))


# In[11]:


clf1 = MultinomialNB().fit(train_count,news_train["label"])


# In[12]:


# Predicting the Test set results, find accuracy
y_pred = clf1.predict(test_count)


# In[13]:


y_pred = pd.DataFrame(y_pred,columns=['label'])
frames=[news_test,y_pred]
result=pd.concat(frames,axis=1)
result


# In[15]:


result.to_csv("news-test.csv", sep='\t')

