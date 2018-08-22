
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import sklearn
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse


# In[2]:


snli_train = pd.read_csv('data/snli-train.csv')
snli_train = snli_train[['sentence1','sentence2','label']]


# In[3]:


snli_dev = pd.read_csv('data/snli-train.csv')
snli_dev = snli_dev[['sentence1','sentence2','label']]


# In[4]:


snli_test = pd.read_csv('data/snli-test.csv',error_bad_lines=False)
snli_test = snli_test[['sentence1','sentence2']]


# In[5]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


# In[6]:


#preprocessing Train data
import re
ps = PorterStemmer()
stops = set(stopwords.words("english"))

clean_train_sentence1=[]
clean_train_sentence2=[]

for review in snli_train["sentence1"]:
    review = re.sub(r"[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    #words = [ps.stem(w) for w in words ]
    clean_train_sentence1.append(" ".join(words))

    
for review in snli_train["sentence2"]:
    review = re.sub(r"[^a-zA-Z]"," ", str(review))
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    #words = [ps.stem(w) for w in words ]
    clean_train_sentence2.append(" ".join(words))


# In[7]:


len(snli_train["sentence1"])


# In[8]:


#preprocessing test data

clean_test_sentence1=[]
clean_test_sentence2=[]

for review in snli_test["sentence1"]:
    review = re.sub("[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    #words = [ps.stem(w) for w in words ]
    clean_test_sentence1.append(" ".join(words))

    

for review in snli_test["sentence2"]:
    review = re.sub("[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    #words = [ps.stem(w) for w in words ]
    clean_test_sentence2.append(" ".join(words))


# In[9]:


#preprocessing dev data

clean_dev_sentence1=[]
clean_dev_sentence2=[]

for review in snli_dev["sentence1"]:
    review = re.sub("[^a-zA-Z]"," ", review)
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    #words = [ps.stem(w) for w in words ]
    clean_dev_sentence1.append(" ".join(words))

    
for review in snli_dev["sentence2"]:
    review = re.sub("[^a-zA-Z]"," ", str(review))
    words = review.lower().split()
    words = [w for w in words if not w in stops]
    #words = [ps.stem(w) for w in words ]
    clean_dev_sentence2.append(" ".join(words))


# In[10]:


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
count_vec1 = TfidfVectorizer(stop_words='english')
count_vec2 = TfidfVectorizer(stop_words='english')


# In[11]:


train_count2=count_vec2.fit_transform(clean_train_sentence2)
train_count1=count_vec1.fit_transform(clean_train_sentence1)
train_count=sparse.hstack((train_count1,train_count2))


# In[12]:


train_count2


# In[13]:


train_count1


# In[14]:


test_count2=count_vec2.transform(clean_test_sentence2)
test_count1=count_vec1.transform(clean_test_sentence1)
test_count=sparse.hstack((test_count1,test_count2))


# In[15]:


dev_count2=count_vec2.transform(clean_dev_sentence2)
dev_count1=count_vec1.transform(clean_dev_sentence2)
dev_count=sparse.hstack((dev_count1,dev_count2))


# In[16]:


from sklearn.cross_validation import KFold,cross_val_score
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
scores=cross_val_score(clf,train_count,snli_train["label"],cv=5,scoring='accuracy')
scores.mean()


# In[17]:



clf1 = MultinomialNB().fit(train_count,snli_train["label"])
y_pred = clf1.predict(test_count)
y_pred = pd.DataFrame(y_pred,columns=['label'])
frames=[snli_test,y_pred]
result=pd.concat(frames,axis=1)
result.to_csv("snli-test.csv", sep='\t')


# In[18]:


y_pred_dev = clf1.predict(dev_count)

