#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:19:12 2022

@author: johndewitt1
V3 created to allow looping of modeling to assess model performance across
multiple instances
"""
#%% normal imports
#import numpy as np
import pandas as pd

# Classifier using urls to predict if a sports website - V1
# Added webpage title (minus stop words)

#%% read in list of urls and yes/no from file - NEW

df_url = pd.read_csv('dataToTrainModel.csv')
## keep yes/no, url
df_url = df_url.loc[:,['goodPage','abrev_domain2','title']]
df_url['combined'] = df_url['abrev_domain2'] + ' ' + df_url['title']
df_positive_url = df_url.loc[df_url.goodPage == 'yes',:]
df_negative_url = df_url.loc[df_url.goodPage == 'no',:]

#%% Classifier proc
def runClassifier(df_positive_url, df_negative_url):
    # function to create input dictionary - URL 
    def createInputURL(df,defString):
        df_list = df['abrev_domain2'].tolist() # convert to list
        output_list = []
        for f in df_list:
            # need to make a dictionary
            temp_dict = dict({f:True})
            temp_list = tuple([temp_dict, defString])
            output_list.append(temp_list)
        #return
        return(output_list)
    
    # function to create list of dictionaries for input to classifier
    def createInput_url_title(df,defString):
        # function to create the list of dictionary entries
        # required by the nlt classifier
        # list entries are [{'word' : True}, 'defString']
        # Assumes a field named 'combined' was created and sent in the dataframe
        df_list = df['combined'].tolist() # convert to list
        # remove NaN
        df_list = [x for x in df_list if pd.isnull(x) == False and x != 'nan']
        output_list = []
        for f in df_list:
            temp_list = tuple([word_feats_url_title(f.lower().strip()), defString])
            #dict_global.update(word_feats_2(f.lower().strip()))
            # for t in temp_list:
            #     output_list.append(t)
            output_list.append(temp_list)
        #return
        return(output_list)
    
    
    # function to create  single word dictinary for a record
    # called by funciton createInput_url_title
    def word_feats_url_title(url_title):
        # function to create dictionary of url and title for input to classifier
        # url is created as the domain name stripped of 'www' and domain extension
        # the title hsould be broken into single words after stripped of special
        # characters and stopwords
        # assumed input is a combined field of url and title
        from nltk.corpus import stopwords
    
        # read in list of stop words
        stopset = list(set(stopwords.words('english')))
        
        import re
        spec_char  = ",.!?/&-:;@'...|-<>$()[]"
    
        # remove special characters
        words_1a = ' '.join(w for w in re.split("["+"\\".join(spec_char)+"]", url_title) if w)
     
        # make 1 word dictionary for each entry
        dict_1 = {}
        dict_1 = dict([(word, True) for word in words_1a.split() if word not in stopset])
    
        return(dict_1) 
    
    # url format for classfier
    pos_feats_url = createInput_url_title(df = df_positive_url, defString = 'positive')
    neg_feats_url = createInput_url_title(df = df_negative_url, defString = 'negative')
    
    # Create test and training sets in the format that the classifier requires -
    # tuple with first item a dictionary of all words in the input string indexed
    # to true, and second item 'positive' or 'negative'
    
    # Create a training set and test set
    from sklearn.model_selection import train_test_split
    train_pos, test_pos = train_test_split(pos_feats_url, test_size =.2)  # test set is 20% of total set
    train_neg, test_neg = train_test_split(neg_feats_url, test_size =.2)  # test set is 20% of total set
    
    trainfeats = train_pos + train_neg  # create single training list
    testfeats = test_pos + test_neg  # create single testing list
    
    # train classifier using the combined training set
    # convert test_pos and test_neg to list of dictionaries only
    def getVar1(inputStr):
        var1,  var2 = inputStr
        return var1
    
    from nltk.classify import NaiveBayesClassifier
    classifier = NaiveBayesClassifier.train(trainfeats)
    test_pos_dict = [(getVar1(f)) for f in test_pos]
    test_neg_dict = [(getVar1(f)) for f in test_neg]
    
    # test classifier
    test_pos_predict = [classifier.classify(f) for f in test_pos_dict]
    test_neg_predict = [classifier.classify(f) for f in test_neg_dict]
    
    # count the occurrences of each prediction
    from collections import Counter
    pos_pred = Counter(test_pos_predict)  # count the true pos and false negs
    neg_pred = Counter(test_neg_predict)  # count the true negs and false pos

    return pos_pred, neg_pred

#%% call clasifier proc
''' Call with positive and negative url
    return pos_pred and neg_pred
'''
out_stats = []
out_stats = pd.DataFrame(columns = ['pos_pred_pos','pos_pred_neg', 'neg_pred_pos', 'neg_pred_neg'])
for index in range(1000):
    pos_pred, neg_pred = runClassifier(df_positive_url, df_negative_url)
    out_stats.loc[index] = [ pos_pred['positive'], pos_pred['negative'], neg_pred['positive'], neg_pred['negative']]
    #out_stats.append([index,pos_pred['positive'], neg_pred])

#%%
total_pos = pos_pred['negative'] + pos_pred['positive']
out_stats['true_pos'] = out_stats['pos_pred_pos']/total_pos
out_stats['false_neg'] = out_stats['pos_pred_neg']/total_pos
total_neg = neg_pred['negative'] + neg_pred['positive']
out_stats['true_neg'] = out_stats['neg_pres_neg']/total_neg
out_stats['false_pos'] = out_stats['neg_pred_pos']/total_neg

out_stats.to_csv('outputStatsExamine2022May19.csv')
#%% compute specificity and sensitivity
total_pos = pos_pred['negative'] + pos_pred['positive']
true_pos = pos_pred['positive']/total_pos
false_neg = pos_pred['negative']/total_pos
total_neg = neg_pred['negative'] + neg_pred['positive']
true_neg = neg_pred['negative']/total_neg
false_pos = neg_pred['positive']/total_neg

# print stats
print('True Positive Rate: {:6.4f}'.format(true_pos))
print('False Negative Rate: {:6.4f}'.format(false_neg))
print('True Negative Rate: {:6.4f}'.format(true_neg))
print('False Positive Rate: {:6.4f}'.format(false_pos))

#%% Examine the predicted vs actual to understand differences

def combineResults(testSet, testPred):
    # combine positive lists test_pos_dict and test_pos_predict into dataframe
    df1 = pd.DataFrame(testSet, columns = ['record', 'actual'])
    df2 = pd.DataFrame(testPred, columns = ['prediction'])
    return pd.concat([df1, df2], axis = 1)

df_pos = combineResults(testSet = test_pos, testPred = test_pos_predict)
df_neg = combineResults(testSet = test_neg, testPred = test_neg_predict)

# write to csv
dd = pd.concat([df_pos, df_neg], axis = 0)
dd.to_csv('outputExamine2022May18.csv')

for i in range(5):
    print(i)
    