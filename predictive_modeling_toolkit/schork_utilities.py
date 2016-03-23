import csv
from pandas import *
from scipy.stats import mode
from sklearn import datasets, linear_model

'''PassengerId	Survived	Pclass	Name					Sex	  Age SibSp Parch	Ticket	Fare Cabin	Embarked
	1			0			3		Braund, Mr. Owen Harris	male	22	1	0	A/5 21171	7.25		S
'''

def read_CSV(file_name):
	open_file = open(file_name, 'rb')
	reader = csv.reader(open_file,dialect='excel')
	return reader

train = DataFrame.from_csv('train.csv',index_col=False)
train['train_ind'] = 1
test = DataFrame.from_csv('test.csv',index_col=False)
test['train_ind'] = 0
master = concat([train,test])

mode_embarked = mode(master['Embarked'])[0][0]
master['Embarked'] = master['Embarked'].fillna(mode_embarked)

master['Gender'] = master['Sex'].map({'female': 0, 'male': 1}).astype(int)
master['FamilySize'] = master['Parch'] + master['SibSp'] + 1
master = concat([master, get_dummies(master['Embarked'], prefix='Embarked')], axis=1)

aggregations = {
    'Ticket': { # work on the "duration" column
        'number_ppl': 'count',  # get the sum, and call this result 'total_duration'
    },
    'Age': {     # Now work on the "date" column
        'max_age': 'max',   # Find the max, call the result "max_date"
        'min_age': 'min',
    },
}

new = master[['Ticket','Age']].groupby('Ticket').agg(aggregations)

#lm_age = linear_model.LinearRegression.fit(master[['Gender','Embarked_S','Embarked_C','Embarked_Q','Fare']])


print new
