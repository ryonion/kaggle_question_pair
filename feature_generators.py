from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
from nltk import pos_tag
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

pos_idx = dict(ADJ=0, ADP=1, ADV=2, CONJ=3, DET=4, NOUN=5, NUM=6, PRT=7, PRON=8, VERB=9, X=10)


stops = set(stopwords.words("english"))
cv = pickle.load(open("models/count_vectorizer", 'rb'))
tfidf = pickle.load(open("models/tfidf_transformer", 'rb'))
lr_model = pickle.load(open("models/lr_categorizer", 'rb'))
w2v = Word2Vec.load("data/emb_tr")
vec_size = w2v.vector_size
    
def shared_percentage(raw_row):
    s1 = raw_row["question1"]
    s2 = raw_row["question2"]
    t_1 = word_tokenize(s1)
    t_2 = word_tokenize(s2)
    
    if len(t_1) > len(t_2):
        t_1, t_2 = t_2, t_1
    sum_shared = 0
    for word in t_1:
        if not word in stops and word in t_2:
            sum_shared += 1
    ret = sum_shared*2/ (len(t_1) + len(t_2))
    return ret

def longest_common_substr_prop(raw_row):
    s1 = raw_row["question1"]
    s2 = raw_row["question2"]
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
       for y in range(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
    return len(s1[x_longest - longest: x_longest])*2 / len(s1+s2)

def is_first_word_identical(row):
    s1 = row["question1"]
    s2 = row["question2"]
    if s1 and s2:
        return int(s1.split()[0] ==  s2.split()[0] )
    return 0

def is_in_same_cat(row):

    q1_counts = cv.transform([row["question1"]])
    q1_tfidf = tfidf.transform(q1_counts)
    q2_counts = cv.transform([row["question2"]])
    q2_tfidf = tfidf.transform(q2_counts)

    q1_cat = lr_model.predict(q1_tfidf)
    q2_cat = lr_model.predict(q2_tfidf)
    return int(q1_cat == q2_cat)

def parts_of_speech(s):
        if type(s) == str:                                                 
                tokens = word_tokenize(s)
        else:
                tokens = s                                                 
        tokens_and_tags = pos_tag(tokens, tagset = "universal")
        return(tokens_and_tags)

def sent_embedding(pos_li, vec_size):
    pos_groups = [ np.zeros((vec_size)) for i in range(len(pos_idx)) ]
    for word, pos in pos_li:
        if pos != '.':
            if word in w2v.wv:
                pos_groups[pos_idx[pos]] += w2v.wv[word]
    return pos_groups

def sent_cos_similarity(pos_group1, pos_group2):
    ret = []
    for i, _ in enumerate(pos_group1):
        ret.append((cosine_similarity(pos_group1[i].reshape(1, -1), pos_group2[i].reshape(1, -1)))[0][0])
    return tuple(ret)
                
# return a vector represents the similarity of two sentence
def sent_similarity(row):
    s1 = row["question1"]
    s2 = row["question2"]
    
    pos_li_1 = parts_of_speech(s1)
    pos_li_2 = parts_of_speech(s2)
    
    # assign the words in a sentence into each of the 11 POS groups
    # substitute each word in each POS group with the word's embedding vector
    # sum the vectors in each POS group of a sentence
    # so we will have a list of 11 vectors for each sentence
    
    emb_list_1 = sent_embedding(pos_li_1, vec_size)
    emb_list_2 = sent_embedding(pos_li_2, vec_size)
    # compute the cosine_similarity of the corresponding groups in the two sentences
    # so we will have a list of 11 cosine_similarity measures.
    cos_sim = sent_cos_similarity(emb_list_1, emb_list_2)
    return sum(cos_sim) / len(pos_idx)

def sent_similarity_group(row):
    s1 = row["question1"]
    s2 = row["question2"]
    
    pos_li_1 = parts_of_speech(s1)
    pos_li_2 = parts_of_speech(s2)
    
    # assign the words in a sentence into each of the 11 POS groups
    # substitute each word in each POS group with the word's embedding vector
    # sum the vectors in each POS group of a sentence
    # so we will have a list of 11 vectors for each sentence
    
    emb_list_1 = sent_embedding(pos_li_1, vec_size)
    emb_list_2 = sent_embedding(pos_li_2, vec_size)
    # compute the cosine_similarity of the corresponding groups in the two sentences
    # so we will have a list of 11 cosine_similarity measures.
    
    return sent_cos_similarity(emb_list_1, emb_list_2)
