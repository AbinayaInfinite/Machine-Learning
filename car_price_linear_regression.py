import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Import the data set and take care of the null values '?'
column_names = ['symboling','normalized_losses','make','fuel_type','aspiration','num_of_doors',
                'body_style','drive_wheels','engine_location','wheel_base','length','width','height',
                'curb_weight','engine_type','num_of_cylinders','engine_size','fuel_system','bore','stroke',
                'compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']
dataset = pd.read_csv('car_price_data.csv', header=None, names = column_names, na_values='?')

print(dataset.head())
print(dataset[dataset.columns[dataset.isnull().any()].tolist()].isnull().sum())

# Remove unused columns
dataset.drop(['symboling','normalized_losses'], axis=1, inplace=True)
print(dataset.head())

print(dataset[dataset.num_of_doors.isnull()])

print(dataset.num_of_doors[dataset.body_style=='sedan'].value_counts())

dataset.loc[27,'num_of_doors'] = 'four'
dataset.loc[63,'num_of_doors'] = 'four'

print(dataset[dataset.bore.isnull()])

dataset.bore.fillna(dataset.bore.mean(), inplace=True)

print(dataset[dataset.stroke.isnull()])
dataset.stroke.fillna(dataset.stroke.mean(), inplace=True)
print(dataset[dataset.horsepower.isnull()])
dataset.horsepower.fillna(dataset.horsepower.mean(), inplace=True)
print(dataset[dataset.peak_rpm.isnull()])
dataset.peak_rpm.fillna(dataset.peak_rpm.mean(), inplace=True)
print(dataset[dataset.price.isnull()])
dataset.drop(dataset[dataset.price.isnull()].index, axis=0, inplace=True)
dataset.head()

print(dataset[dataset.columns[dataset.isnull().any()].tolist()].isnull().sum())

# Deal the categorical values

print(dataset.num_of_cylinders.value_counts())

dataset.loc[dataset.index[dataset.num_of_cylinders == 'four'], 'num_of_cylinders'] = 4
dataset.loc[dataset.index[dataset.num_of_cylinders == 'five'], 'num_of_cylinders'] = 5
dataset.loc[dataset.index[dataset.num_of_cylinders == 'six'], 'num_of_cylinders'] = 6
dataset.loc[dataset.index[dataset.num_of_cylinders == 'eight'], 'num_of_cylinders'] = 8
dataset.loc[dataset.index[dataset.num_of_cylinders == 'two'], 'num_of_cylinders'] = 2
dataset.loc[dataset.index[dataset.num_of_cylinders == 'three'], 'num_of_cylinders'] = 3
dataset.loc[dataset.index[dataset.num_of_cylinders == 'twelve'], 'num_of_cylinders'] = 12

dataset.num_of_cylinders = dataset.num_of_cylinders.astype('int')

print(dataset.dtypes)

cat_columns = ['make','fuel_type', 'fuel_system', 'aspiration', 'num_of_doors', 'body_style',
               'drive_wheels','engine_location','engine_type']

df = pd.get_dummies(dataset, columns= cat_columns, drop_first=True)
print(df.head())

# Split the train and test data
y = df.price
x = df.drop(['price'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 1)

# Fit and predict

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

r_square = r2_score(y_test, y_pred)
print(r_square)

actual_data = np.array(y_test)

for i in range(len(y_pred)):
    actual = actual_data[i]
    predicted = y_pred[i]
    explained = ((actual_data[i] - y_pred[i])/actual_data[i])*100
    print('Actual value ${:,.2f}, Predicted value ${:,.2f} (%{:.2f})'.format(actual, predicted, explained))
    
    