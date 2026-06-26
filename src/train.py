from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(df):
    X = df.drop('survived', axis = 1)
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth = 4,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions