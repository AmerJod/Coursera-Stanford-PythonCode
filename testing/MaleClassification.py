from sklearn import tree

X = [[181,70,44],[177,70,43],[160,60,38],[154,54,37],[175,64,39]]
Y = ['male','female','female','female','male']


clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[190,65,70]])

print(prediction)