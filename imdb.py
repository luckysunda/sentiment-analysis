
# coding: utf-8

# In[1]:


import sklearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test set
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.naive_bayes import MultinomialNB


# In[2]:


movie_train = pd.read_csv('data/imdb-train.csv')
movie_train = movie_train[['review','rating']]


# In[3]:


movie_test = pd.read_csv('data/imdb-test.csv',error_bad_lines=False)
movie_test = movie_test[['review']]


# In[4]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


# In[5]:


movie_train


# In[6]:


import re
ps = PorterStemmer()
stops = set(stopwords.words("english"))

clean_train_reviews=[]

for review in movie_train["review"]:
    review = re.sub("[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    words = [ps.stem(w) for w in words ]
    clean_train_reviews.append(" ".join(words))
  


# In[7]:


clean_test_reviews=[]

for review in movie_test["review"]:
    review = re.sub("[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    words = [ps.stem(w) for w in words ]
    clean_test_reviews.append(" ".join(words))
    


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
count_vec = TfidfVectorizer(stop_words='english')
train_count=count_vec.fit_transform(clean_train_reviews)
test_count=count_vec.transform(clean_test_reviews)


# In[9]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
from sklearn.cross_validation import KFold,cross_val_score
scores=cross_val_score(clf,train_count,movie_train["rating"],cv=10,scoring='accuracy')


# In[10]:


scores.mean()


# In[11]:


clf1 = MultinomialNB().fit(train_count,movie_train["rating"])


# In[12]:


# Predicting the Test set results, find accuracy
y_pred = clf1.predict(test_count)
y_pred


# In[13]:



y_pred = pd.DataFrame(y_pred,columns=['rating'])
frames=[movie_test,y_pred]
result=pd.concat(frames,axis=1)
result.to_csv("imdb-test.csv", sep='\t')

