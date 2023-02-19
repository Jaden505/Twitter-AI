from data import Data

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC


class Models:
    def __init__(self):
        pass

    def naive_bayes(self):
        # Convert the text data into a matrix of token counts
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        # Train a Naive Bayes classifier on the training data
        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)

        # Convert the test data into a matrix of token counts
        X_test_vec = vectorizer.transform(X_test)

        # Evaluate the accuracy of the classifier on the test data
        accuracy = clf.score(X_test_vec, y_test)
        print("Accuracy:", accuracy)

    def svm(self):
        # Convert the text data into a matrix of TF-IDF features
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        # Train an SVM classifier on the training data
        clf = SVC(kernel='linear')
        clf.fit(X_train_vec, y_train)

        # Convert the test data into a matrix of TF-IDF features
        X_test_vec = vectorizer.transform(X_test)

        # Evaluate the accuracy of the classifier on the test data
        accuracy = clf.score(X_test_vec, y_test)
        print("Accuracy:", accuracy)


if __name__ == '__main__':
    d = Data()
    d.get_local_data()
    X_train, X_test, y_train, y_test = d.shape_data()

    m = Models()
    m.svm()
    m.naive_bayes()
