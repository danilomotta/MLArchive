from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_breast_cancer

from ml_archive import MLArchive

if __name__ == "__main__":
    arch = MLArchive()
    X, y = load_breast_cancer(return_X_y=True)

    names = ["Neural Net", "Log reg", "Random Forest", "Decision Tree"]

    classifiers = [
        Pipeline([('scaler', RobustScaler()),
                  ('mlp', MLPClassifier(alpha=0.5,
                                        hidden_layer_sizes=(64,4),
                                        activation='relu',
                                        max_iter=100000,
                                        validation_fraction=0.3,
                                        early_stopping=True, 
                                        random_state=42))]),
        LogisticRegression(C=0.0001,max_iter=10000, 
                     intercept_scaling=False,
                     fit_intercept=False),
        RandomForestClassifier(max_depth=8, n_estimators=50, 
                               max_samples=0.7, random_state=42),
        DecisionTreeClassifier(max_depth=16),]
    
    # iterate over classifiers
    print('------------------------------------------------')
    for name, clf in zip(names, classifiers):
        cv_results = cross_validate(clf, X, y, cv=5,
                                    return_train_score=True)
        print(name + ' train: ' + ('%.2f' % cv_results['train_score'].mean()).lstrip('0'))
        print(name + ' test: ' + ('%.2f' % cv_results['test_score'].mean()).lstrip('0'))

        arch.save_model(clf, metric='F1', 
            train_res=cv_results['train_score'].mean(), 
            test_res=cv_results['test_score'].mean())
        print('------------------------------------------------')
    arch.save_archive('MLModel.arch')
    print(arch.get_ranked_models())
