import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
import csv
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import pos_tag
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models import Word2Vec
from importlib import reload

import feature_generators as fg

pos_idx_inv = ["ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRT", "PRON", "VERB", "X"]

reload(fg)


stops = set(stopwords.words("english"))
##cleaned_raw_file = "data/cleaned_raw.csv"

def extend_abbre(question):
    question = question.lower().replace("won't","will not").replace("n't","not").replace("it's","it is") \
    .replace("'ve"," have").replace("i'm","i am").replace("'re"," are").replace("he's","he is").replace("she's","she is") \
    .replace("'s"," own").replace("what's","what is").replace("+","plus").replace("'ll"," will").replace("'d"," would") \
    .replace("#","sharp").replace("="," equal").replace(",000","k").replace("&","and").replace("|","or")
    return question

def stemmer(train_data):
    cp_tr = train_data.copy()
    stem_tool = SnowballStemmer('english')
    cp_tr['question1'] = cp_tr.question1.map(lambda x: ' '.join([stem_tool.stem(word.lower()) for word in word_tokenize(extend_abbre(str(x))) ]))
    cp_tr['question2'] = cp_tr.question2.map(lambda x: ' '.join([stem_tool.stem(word.lower()) for word in word_tokenize(extend_abbre(str(x))) ]))
    return cp_tr

def preprocess(data_needed, start_idx, raw_file, cleaned_raw_file):
    f, is_new_file = to_append(cleaned_raw_file, data_needed)
    if f:
        last_idx = data_needed-1
        raw = pd.read_csv(raw_file, nrows = last_idx+1)
        to_clean = raw.loc[start_idx:last_idx]
        cleaned = stemmer(to_clean)
        cleaned.to_csv(f, header=is_new_file, index=False)
    print("Done pre-processing %d data."%(data_needed))

def to_append(filename, data_needed):
    try:
        df = pd.read_csv(filename)
        if df.shape[0] >= data_needed:
            return 0, 0
        return open(filename, 'a', encoding = "utf-8"), False
    except: 
        return open(filename, 'w', encoding = "utf-8"), True

def gen_ready_mat(data_needed, start_idx, cleaned_raw_file, ready_mat_file, feature_functions_list):
    
    f, is_new_file = to_append(ready_mat_file, data_needed)
    if not is_new_file:
        raw = pd.read_csv(cleaned_raw_file, nrows = start_idx+1)
        ready_mat = pd.read_csv(ready_mat_file)
        
        for f_name in feature_functions_list:
            try:
                ready_mat[get_method_abbre(f_name)]
                
            # if the feature has never been added
            except KeyError:

                if f_name == "sent_similarity_group":
                    continue
                
                print("Feature missing:", f_name)
                new_feature_col = raw.apply(  eval("fg."+f_name), axis=1, raw=True )
                try:
                    for r_id, row in enumerate(new_feature_col):
                        for num_id, num in enumerate(row):
                            ready_mat.loc[r_id, pos_idx_inv[num_id] ] = num
                except:
                    ready_mat[get_method_abbre(f_name)] = raw.apply(  eval("fg."+f_name), axis=1, raw=True )
                ready_mat.to_csv(ready_mat_file, header=True, index=False)
                f, _ = to_append(ready_mat_file, data_needed)
        
    if f:
        last_idx = data_needed-1
        raw = pd.read_csv(cleaned_raw_file, nrows = last_idx+1)
        
        # select rows to be used
        raw_cols = raw.loc[start_idx+1:last_idx]

        
        # apply feature methods to newly added rows in ready_mat
        for f_name in feature_functions_list:
            new_feature_col = raw_cols.apply( eval("fg."+f_name), axis=1, raw=True )
            try:
                for r_id, row in enumerate(new_feature_col):
                                   
                    for num_id, num in enumerate(row):
                        raw_cols.loc[start_idx+r_id-1, pos_idx_inv[num_id] ] = num
            except:
                raw_cols[get_method_abbre(f_name)] = new_feature_col
        
        raw_cols = raw_cols.drop(['id', 'question1', 'question2'], axis=1)
        raw_cols.to_csv(f, header=is_new_file, index=False)
        
    print("Done generating %d ready data."%(data_needed))

def get_method_abbre(method_name):
    ret = ""
    for i in method_name.split('_'):
        ret += i[0]
    return ret

# apply feature function to the raw file and get a new feature col
def add_feature(feature_func, cleaned_raw_file, ready_mat_file, feature_name):
    raw = pd.read_csv(cleaned_raw_file)
    new_col = raw.apply(feature_func, axis=1, raw=True)
    mat = pd.read_csv(ready_mat_file)
    mat[feature_name] = new_col
    mat.to_csv(ready_mat_file, index=False)
    print("feature: %s added"%(feature_name))
    
def f_drop(feat_name, file_name):
    try:
        df = pd.read_csv(file_name)
        df = df.drop([feat_name],axis = 1)
        df.to_csv(file_name, index = False)
    except ValueError:
        print('not exist.')

def pd_f(file_name):
    print(pd.read_csv(file_name))

def pd_h(file_name, row):
    print(pd.read_csv(file_name).head(row))
