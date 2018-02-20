
# coding: utf-8

# ### tools

# In[1]:

import numpy as np
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegression as lr
from sklearn.feature_extraction.text import CountVectorizer
from nltk import FreqDist
from nltk.corpus import stopwords


# In[2]:

def dif_length(line1, line2):
    return abs(len(line1) - len(line2))


# In[ ]:



