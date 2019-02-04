#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The :mod:`autostopwordgen.generator` module implements a research paper published by Olatunde et al.,
https://www.researchgate.net/publication/318969652_AN_AUTO-GENERATED_APPROACH_OF_STOP_WORDS_USING_AGGREGATED_ANALYSIS

Created on Sat Feb  2 20:50:34 2019
@author: Nagaraju Budigam
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class AutoStopWordsGen:

    def __init__(self, corpus):
        """
        :param corpus: corpus
        """
        logging.info('initializing the AutoStopwordsGen started')
        cv = CountVectorizer()
        cvft = cv.fit_transform(corpus)
        tfidfcv = TfidfVectorizer()
        tfidfcvft = tfidfcv.fit_transform(corpus)
        self.tfidfcv=tfidfcv
        self.cvft=cvft
        logging.info('initializing the AutoStopwordsGen completed')

    def get_stopwords(self, top_n=.95, last_n=.1):

        """
        :param top_n: top n percent threshold
        :param last_n: last n percent threshold
        :return: returns stopwords
        """
        logging.info('generating stopwords started')
        word_freq_df = pd.DataFrame({'word': self.tfidfcv.get_feature_names(),
                                     'frequency':np.asarray(self.cvft.sum(axis=0)).ravel().tolist(),
                                     'idf':self.tfidfcv.idf_})
        word_freq_df.sort_values(by=['frequency'],ascending = False,inplace=True)

        word_freq_df['prob']=word_freq_df.frequency/word_freq_df.shape[0]
        word_freq_df['entropy']=word_freq_df.prob.apply(lambda x: x*np.log(1/x))
        word_freq_df['vp']=np.power(word_freq_df.prob-word_freq_df.prob.mean(),2)/word_freq_df.shape[0]

        stopwords=dict({'frequency':[],'idf':[],'entropy':[],'vp':[]})
        cols=['frequency','entropy','vp']

        for col in cols:
            # print(col,' : ',word_freq_df[col].quantile([last_n,top_n]).tolist())
            if(col=='frequency'):
                top_5_percent=word_freq_df[col].quantile([last_n,top_n]).tolist()[1]
                stopwords[col]=word_freq_df[word_freq_df[col]>=top_5_percent].word.tolist()
            else:
                last_10_percent=word_freq_df[col].quantile([last_n,top_n]).tolist()[0]
                stopwords[col]=word_freq_df[word_freq_df[col]<=last_10_percent].word.tolist()

        for key in stopwords.keys():
            stopwords[key]=set(stopwords[key])
        very_high_aggregation=word_freq_df[word_freq_df.frequency<2].word.tolist()
        very_high_aggregation.extend(list(stopwords['frequency'].intersection(stopwords['entropy']).intersection(stopwords['vp'])))
        very_high_aggregation=list(set(very_high_aggregation))
        logging.info('# stopwords generated :: '+str(len(very_high_aggregation)))
        logging.info('generating stopwords completed')
        return very_high_aggregation
