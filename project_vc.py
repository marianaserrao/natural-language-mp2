import numpy as np
import re, string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

train_path="data/train.txt"
test_path="data/test.txt"
output_path="results.txt"
labels = ["Poor", "Unsatisfactory", "Good", "VeryGood", "Excellent"]

wl = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
def lemmatizer(sentence):
    word_pos_tags = nltk.pos_tag(word_tokenize(sentence))
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]
    return " ".join(a)

def pre_process(sentence):
  #sentence = sentence.lower() 
  #sentence=sentence.strip()
  #sentence=re.compile('<.?>').sub('', sentence) 
  sentence = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', sentence)
  #sentence = re.sub('\s+', ' ', sentence)
  #sentence = re.sub(r'[[0-9]]',' ',sentence) 
  sentence=re.sub(r'[^\w\s]', '', str(sentence).lower().strip())
  sentence = re.sub(r'\d',' ',sentence) 
  #sentence = re.sub(r'\s+',' ',sentence) 
  # remove line feed and tab characters
  #sentence = re.sub(r'[\n\t]', '', sentence)

  output = lemmatizer(sentence)

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