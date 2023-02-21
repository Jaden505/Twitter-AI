from data import Data

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

class Models:
    def naive_bayes(self):
        # Convert the text data into a matrix of token counts
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)

        X_test_vec = vectorizer.transform(X_test)

        # Evaluate the accuracy of the classifier on the test data
        accuracy = clf.score(X_test_vec, y_test)
        print("Accuracy:", accuracy)

        pickle.dump(clf, open('models/nb_model.sav', 'wb'))

    def random_forest(self):
        # Convert the text data into a matrix of word counts
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train_vec, y_train)

        X_test_vec = vectorizer.transform(X_test)

        # Evaluate the accuracy of the classifier on the test data
        accuracy = clf.score(X_test_vec, y_test)
        print("Accuracy:", accuracy)

        pickle.dump(clf, open('models/rf_model.sav', 'wb'))

if __name__ == '__main__':
    d = Data()
    d.get_local_data()
    X_train, X_test, y_train, y_test = d.shape_data()

    # m.naive_bayes()
    # m.random_forest()
