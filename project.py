import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

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

# covert strings to numerical feature vectors (bag of words)
count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(data)
# apply tf-idf
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X)

# grid of parameters for optimization search
alpha_opt = np.arange(0.05,1.01,0.05)
grid = {
  "alpha": alpha_opt,
  "fit_prior": [True, False]
}

# grid search for best parameters
clf_gs = GridSearchCV(estimator=MultinomialNB(), param_grid=grid, cv= 5)

# train classifier
clf = clf_gs.fit(X, target)
print(clf.best_score_)

# split data into train and test datasets
# X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=0)

# clf = clf_gs.fit(X_train, y_train)
# inf = clf.predict(X_test)
# report = classification_report(inf, y_test)
# print(report)