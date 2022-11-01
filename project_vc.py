import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

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
  # output = ps.stem(output)
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
  ('vc', VotingClassifier(
    estimators=[
      ('mnb', mnb), 
      ('lr', lr), 
      ('svc', svc)
    ], 
    voting='hard'
  ))
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