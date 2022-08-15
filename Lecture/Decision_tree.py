# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
df = sns.load_dataset('titanic')
df.head()

df.tail()

df_x = df[['sex','pclass','fare']]
df_y = df['survived']

df_x = pd.get_dummies(df_x, drop_first=True)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_x,df_y,random_state=1)

from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=2, random_state=1)

model.fit(train_x, train_y)

model.predict(test_x)

model.score(test_x,test_y)

from sklearn.tree import plot_tree
plot_tree(model, feature_names=train_x.columns, class_names=True, filled=True)


def my_desicion_tree(dfx, dfy):
    model = tree.DecisionTreeClassifier(max_depth=2, random_state=1)
    model.fit(dfx, dfy)
    plot_tree(model, feature_names=dfx.columns, class_names=True, filled=True) 


test = pd.read_csv('test.csv')

test

test_x = test[['sex','type']]
test_y = test['purchase']

test_x

test_x = pd.get_dummies(test_x, drop_first=True)

test_x

test_y

test_y = pd.get_dummies(test_y, drop_first=True)

test_y

my_desicion_tree(test_x, test_y)
