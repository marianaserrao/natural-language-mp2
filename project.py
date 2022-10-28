import numpy as np
import re
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

data_path="data/train.txt"
labels = ["Poor", "Unsatisfactory", "Good", "VeryGood", "Excellent"]

def pre_process(sentence):
  # remove line feed and tab characters
  output = re.sub(r'[\n\t]', '', sentence)

  return output

with open(data_path) as f:
    raw_data = f.readlines()

data = [pre_process(line.split('=')[2]) for line in raw_data]
target = [labels.index(line.split('=')[1]) for line in raw_data]

# pipeline of transformers and estimator
clf_pipe = Pipeline([
  ('tfidf', TfidfVectorizer( # covert strings to numerical feature vectors (tf-idf)
    use_idf=False, # idf(t) = 1
    ngram_range=(1,4), # extracts up to 4-grams
    lowercase=True,
  )), 
  ('clf', MultinomialNB(
    fit_prior=False, # use a uniform prior
    alpha = 0.75 # additive smoothing parameter
  )),
  # ('clf', LogisticRegression()),
])

"""
# grid of parameters for optimization search
alpha_opt = np.arange(0.05,1.01,0.05)
ngram_opt = [(1,n) for n in list(range(1, 7))]
grid = {
  "tfidf__use_idf": [True, False],
  "tfidf__stop_words": [None, stopwords.words('english')],
  "tfidf__ngram_range": ngram_opt,
  "tfidf__lowercase" : [True, False],
  "clf__fit_prior": [True, False],
  "clf__alpha": alpha_opt,
}

# grid search for best parameters
clf_gs = GridSearchCV(clf_pipe, grid, cv= 5)

# train classifier with grid search
clf = clf_gs.fit(data, target)
print(clf.best_score_,)
"""

# train classifier
clf = clf_pipe.fit(data, target)
# get cross validation scores
scores = cross_val_score(clf_pipe, data, target, cv=5)
print(scores, np.mean(scores))