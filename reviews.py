import re, string
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

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
  wl = WordNetLemmatizer()
  word_pos_tags = nltk.pos_tag(word_tokenize(sentence))
  a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]
  return " ".join(a)

def pre_process(sentence):
  sentence = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', sentence)
  sentence=re.sub(r'[^\w\s]', '', str(sentence).lower().strip())
  sentence = re.sub(r'\d',' ',sentence) 
  # remove line feed and tab characters
  sentence = re.sub(r'[\n\t]', '', sentence)

  output = lemmatizer(sentence)

  return output

def build_clf_pipeline():
  #estimators
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

  return clf_pipe

def main():
  # initialize parser
  parser = argparse.ArgumentParser()

  # add expected arguments
  parser.add_argument("-test", "--Test")
  parser.add_argument("-train", "--Train")

  # read arguments
  args = parser.parse_args()
  train_path=args.Train
  test_path=args.Test

  labels = ["Poor", "Unsatisfactory", "Good", "VeryGood", "Excellent"]

  # get data arrrays
  with open(train_path) as f:
      train_data = f.readlines()
  with open(test_path) as f:
      test_data = f.readlines()

  # format data and targets
  X_train = [pre_process(line.split('=')[2]) for line in train_data]
  y_train = [labels.index(line.split('=')[1]) for line in train_data]
  X_test = [pre_process(line) for line in test_data]

  clf_pipe = build_clf_pipeline()

  # train classifier
  clf = clf_pipe.fit(X_train, y_train)

  # prediction
  pred = clf.predict(X_test)
  for p in pred: print(f'={labels[p]}=')

if __name__ == "__main__":
    main()