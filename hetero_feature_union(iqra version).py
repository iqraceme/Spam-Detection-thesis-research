"""
=============================================
Feature Union with Heterogeneous Data Sources
=============================================

Datasets can often contain components of that require different feature
extraction and processing pipelines.  This scenario might occur when:

1. Your dataset consists of heterogeneous data types (e.g. raster images and
   text captions)
2. Your dataset is stored in a Pandas DataFrame and different columns
   require different processing pipelines.

This example demonstrates how to use
:class:`sklearn.feature_extraction.FeatureUnion` on a dataset containing
different types of features.  We use the 20-newsgroups dataset and compute
standard bag-of-words features for the subject line and body in separate
pipelines as well as ad hoc features on the body. We combine them (with
weights) using a FeatureUnion and finally train a classifier on the combined
set of features.

The choice of features is not particularly helpful, but serves to illustrate
the technique.
"""

# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
sms = pd.read_csv('C:\\Users\\Hamza\\Desktop\\spam.csv', encoding='latin-1')
print(sms.head())

sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
sms = sms.rename(columns = {'v1':'label','v2':'message'})

print(sms.head(10))
data_train, data_test, y_train, y_true = \
train_test_split(sms['message'], sms['label'], test_size=0.2) 

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, message):
        return [{'length': len(text)
                 }
                for text in message]


class SubjectBodyExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, message):
        features = np.recarray(shape=(len(message),),
                               dtype=[ ('message', object)])
        for i, text in enumerate(message):
            bod = text
            features['message'][i] = bod

#            prefix = 'Subject:'
#            sub = ''
#            for line in headers.split('\n'):
#                if line.startswith(prefix):
#                    sub = line[len(prefix):]
#                    break
#            features['subject'][i] = sub

        return features


pipeline = Pipeline([
    # Extract the subject & body
    ('message', SubjectBodyExtractor()),

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            ('message', Pipeline([
                ('selector', ItemSelector(key='message')),
                ('count', CountVectorizer(ngram_range=(2, 2), analyzer='word')),
            ])),


            # Pipeline for pulling ad hoc features from post's body
            ('body_stats', Pipeline([
                ('selector', ItemSelector(key='message')),
                ('stats', TextStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

        ],

        # weight components in FeatureUnion
        
    )),

    # Use a SVC classifier on the combined features
    ('svc', SVC(kernel='linear')),
])

# limit the list of categories to make running this example faster.
#==============================================================================
# categories = ['alt.atheism', 'talk.religion.misc']
# train = fetch_20newsgroups(random_state=1,
#                            subset='train',
#                            categories=categories,
#                            )
# test = fetch_20newsgroups(random_state=1,
#                           subset='test',
#                           categories=categories,
#                           )
#==============================================================================

pipeline.fit(data_train, y_train)
y = pipeline.predict(data_test)
print(classification_report(y, y_true))
