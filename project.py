import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


data_path="data/train.txt"
labels = ["Poor", "Unsatisfactory", "Good", "VeryGood", "Excellent"]

def pre_process(sentence):
  # ps = PorterStemmer() #stemming
  lemmatizer = WordNetLemmatizer()
  # remove line feed and tab characters
  inter = re.sub(r'[\n\t]', '', sentence)
  # output = ps.stem(inter)
  output = lemmatizer.lemmatize(inter)

  return output

with open(data_path) as f:
    raw_data = f.readlines()

data = [pre_process(line.split('=')[2]) for line in raw_data]
target = [labels.index(line.split('=')[1]) for line in raw_data]

# # covert strings to numerical feature vectors (bag of words)
# count_vectorizer = CountVectorizer()
# X = count_vectorizer.fit_transform(data)
# # apply tf-idf
# tfidf_transformer = TfidfTransformer()
# X = tfidf_transformer.fit_transform(X)

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