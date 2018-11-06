# While not a beautiful example of the power of KS,
# here is an example of the KS viz in action

from ks_plot import make_ks_plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                             random_state=42)
clf.fit(X_train, y_train)

train_proba = clf.predict(X_train)
test_proba = clf.predict(X_test)

make_ks_plot(y_train, train_proba, y_test, test_proba)
