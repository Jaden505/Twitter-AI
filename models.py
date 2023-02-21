from data import Data

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

class Models:
    def naive_bayes(self):  # 0.78 accuracy
        # Convert the text data into a matrix of token counts
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)

        X_test_vec = vectorizer.transform(X_test)

        self.evaluate(clf, X_test_vec, y_test)

        pickle.dump(clf, open('models/nb_model.sav', 'wb'))

    def random_forest(self):  # 0.63 accuracy
        # Convert the text data into a matrix of word counts
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train_vec, y_train)

        X_test_vec = vectorizer.transform(X_test)

        self.evaluate(clf, X_test_vec, y_test)

        pickle.dump(clf, open('models/rf_model.sav', 'wb'))

    def support_vector_machine(self):  # 0.73 accuracy
        sgd = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf',
                         SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                        ])
        sgd.fit(X_train, y_train)

        y_pred = sgd.predict(X_test)

        print('accuracy %s' % accuracy_score(y_pred, y_test))

        # pickle.dump(sgd, open('models/svm_model.sav', 'wb'))

    def evaluate(self, model, X, y):
        # Evaluate the accuracy of the classifier on the test data
        accuracy = model.score(X, y)
        print("Accuracy:", accuracy)

if __name__ == '__main__':
    d = Data()
    d.get_local_data()
    X_train, X_test, y_train, y_test = d.shape_data()

    # data = np.load('trainingandtestdata/shaped_data.npz', allow_pickle=True)
    # X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    m = Models()
    m.support_vector_machine()
    # m.naive_bayes()
    # m.random_forest()
