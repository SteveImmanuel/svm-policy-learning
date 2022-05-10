from sklearn import svm

# state xa ya za ha la wa cos sin hb wb hc wc
# action px py pz d1x d1y d1z d2x d2y d2z d3x d3y d3z f pre spt1 spt2 spt3

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
