import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from  sklearn.model_selection import learning_curve
from xgboost import XGBClassifier



pd.set_option('display.max_row',111)
pd.set_option('display.max_column',111)

data = pd.read_csv('diabetes.csv')
print(data.head())
print(data.shape)

print(data.isna().sum())

print(data['Outcome'].value_counts())

# plt.figure(figsize=(10,20))
# sns.heatmap(data,cbar=False)
# plt.show()

X = data.drop('Outcome',axis=1)
y = data['Outcome']


print(data.dtypes)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Içi je n'ai que des valeurs int et float donc pas besoin de standardScaler

prepocessor = make_pipeline(SelectKBest(f_classif,k=5))

Adaboost = make_pipeline(prepocessor,AdaBoostClassifier(random_state=0))

Tree = make_pipeline(prepocessor,DecisionTreeClassifier(random_state=0))

SVM = make_pipeline(prepocessor,StandardScaler(),SVC(random_state=0))

Forest = make_pipeline(prepocessor,RandomForestClassifier(random_state=0))

XG = make_pipeline(prepocessor,XGBClassifier(random_state=0))


def evaluation(model) :

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test,y_pred))



list_models = {
    'Adaboost':Adaboost,
    'Tree': Tree,
    'SVM':SVM,
    'Forest':Forest,
    'XG':XG
}

for name,model in list_models.items() :
    print(name)
    evaluation(model)


grid_params = {
    'adaboostclassifier__n_estimators':[50,100,200,500],
    'adaboostclassifier__learning_rate':[0.001,0.01,0.1,1,10],
    
    # 'max_depth':[1,2,3,5],
    # 'min_samples_depth':[2,5,10]
}

grid = GridSearchCV(estimator=Adaboost,param_grid=grid_params,cv=5,scoring='f1')
grid.fit(X_train,y_train)
# print("Les meilleurs params sont :")
# print(grid.best_params_)

model = grid.best_estimator_ 

print(model.score(X_test,y_test))


N,train_score,test_score = learning_curve(estimator=model,X=X_train,y=y_train,cv=5,scoring='f1')
plt.figure(figsize=(10,20))
plt.plot(N,train_score.mean(axis=1))
plt.plot(N,test_score.mean(axis=1))
plt.show()


print("Cette fois , j'ai un score de près de 80% , c'est super !")