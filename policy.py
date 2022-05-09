from sklearn import svm

X = [[0, 0, 1, 4, 3, 12, 43], [1, 1, 5, 23, 54, 12, 5]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print(clf.predict([[.5, .4, .4, .2, .1, .6, .2]]))

clf = svm.OneClassSVM()
clf.fit(X)
print(clf.predict(X))
print(clf.score_samples(X))

clf = svm.SVR()
clf.fit(X, y)
print(clf.predict([[0, 3, 1, 4, 5, 6, 7]]))
