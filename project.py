import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

train_path="data/train.txt"
test_path="data/test.txt"
output_path="results.txt"
labels = ["Poor", "Unsatisfactory", "Good", "VeryGood", "Excellent"]

def pre_process(sentence):
  # ps = PorterStemmer() #stemming
  # lemmatizer = WordNetLemmatizer()
  # remove line feed and tab characters
  output = re.sub(r'[\n\t]', '', sentence)
  # output = ps.stem(inter)
  # output = lemmatizer.lemmatize(output)

  return output

with open(train_path) as f:
    train_data = f.readlines()

with open(test_path) as f:
    test_data = f.readlines()

X_train = [pre_process(line.split('=')[2]) for line in train_data]
y_train = [labels.index(line.split('=')[1]) for line in train_data]
X_test = [pre_process(line) for line in test_data]

#classifiers
mnb = MultinomialNB(
    fit_prior=False, # use a uniform prior
    alpha = 0.75 # additive smoothing parameter
)
gnb = GaussianNB(
  var_smoothing=0.08
)
lr =  LogisticRegression(
  C=2,
  max_iter=110
)
svc = LinearSVC(
  class_weight='balanced', 
  C=0.39
)

# pipeline of transformers and estimator
clf_pipe = Pipeline([
  ('tfidf', TfidfVectorizer( # covert strings to numerical feature vectors (tf-idf)
    use_idf=False, # idf(t) = 1
    ngram_range=(1,4), # extracts up to 4-grams
    lowercase=True,
  )),    
  ('toarray', FunctionTransformer(
    lambda x: x.toarray(), accept_sparse=True
  )),
  ('clf', lr)
])

# train classifier
clf = clf_pipe.fit(X_train, y_train)

# get cross validation scores
scores = cross_val_score(clf_pipe, X_train, y_train, cv=5)
print("Cross Validation Scores:", scores, np.mean(scores))

pred = clf_pipe.predict(test_data)
with open(output_path, "w") as f:
  for p in pred:
    f.write(f'={labels[p]}=\n')

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