import numpy as np
import pandas as pd
import random
import csv
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools import eval_measures
from sklearn.preprocessing import PolynomialFeatures
import sqlite3
import datetime as dt

def split_data(data, prob):
    """input: 
     data: a list of pairs of x,y values
     prob: the fraction of the dataset that will be testing data, typically prob=0.2
     output:
     two lists with training data pairs and testing data pairs 
    """

    train_pairs = []
    test_pairs = []
    for e in data:
        r = random.random()
        if r <= prob:
            test_pairs.append(e)
        else:
            train_pairs.append(e)
    return train_pairs, test_pairs
     
    

def train_test_split(x, y, test_pct):
    """input:
    x: list of x values, y: list of independent values, test_pct: percentage of the data that is testing data=0.2.

    output: x_train, x_test, y_train, y_test lists
    """
    
    pairs = zip(x, y)
    train_pairs, test_pairs = split_data(pairs, test_pct)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in train_pairs:
        x_val, y_val = i
        x_train.append(x_val)
        y_train.append(y_val)
    for i in test_pairs:
        x_val, y_val = i
        x_test.append(x_val)
        y_test.append(y_val)
    return x_train, x_test, y_train, y_test




if __name__=='__main__':

    random.seed(1)
    # Setting p to 0.2 allows for a 80% training and 20% test split
    p = 0.2

    def load_file(file_path):
        """input: file_path: the path to the data file
           output: X: array of independent variables values, y: array of the dependent variable values
        """
        con = sqlite3.connect(file_path)

        df = pd.read_sql('SELECT precipIntensity, windSpeed, visibility, avgSpeed, acc, summary, borough, hourComparator FROM allHours WHERE avgSpeed IS NOT NULL AND precipIntensity IS NOT NULL AND visibility IS NOT NULL;', con)
        con.close()        
        X = df.iloc[:,[0, 1, 2, 3]]
        X = X.subtract(X.mean(axis=0)).divide(X.std(axis=0)).values
        y = df.iloc[:,4].values
        col_length = y.shape[0]
        
        one_hot_weather = np.zeros(shape=(col_length, 3))
        one_hot_borough = np.zeros(shape=(col_length, 4))
        one_hot_time = np.zeros(shape=(col_length, 23))
        one_hot_date = np.zeros(shape=(col_length, 1))
        weather_summary = df.iloc[:,5]
        borough_column = df.iloc[:,6]
        datetime_column = df.iloc[:,7]
        for i in range(col_length):
            weather = weather_summary[i]
            if weather == "Rain" or weather == "Light Rain" or weather == "Drizzle" or weather == "Heavy Rain":
                one_hot_weather[i][0] = 1
            elif weather == "Overcast" or weather == "Partly Cloudy" or weather == "Cloudy" or weather == "Mostly Cloudy" or weather == "Foggy":
                one_hot_weather[i][1] = 1
            elif weather == "Snow" or weather == "Light Snow" or weather == "Flurries":
                one_hot_weather[i][2] = 1

            borough = borough_column[i]
            if borough == "BRONX":
                one_hot_borough[i][0] = 1
            elif borough == "BROOKLYN":
                one_hot_borough[i][1] = 1
            elif borough == "MANHATTAN":
                one_hot_borough[i][2] = 1
            elif borough == "QUEENS":
                one_hot_borough[i][3] = 1

            year, month, day, hour = datetime_column[i].split()
            datetime = dt.datetime(int(year), int(month), int(day), int(hour))

            if datetime.hour != 23:
                one_hot_time[i][datetime.hour] = 1

            if 0 <= datetime.weekday() <= 4:
                one_hot_date[i][0] = 1


        X = np.append(X, one_hot_weather, axis=1)
        X = np.append(X, one_hot_borough, axis=1)
        X = np.append(X, one_hot_time, axis=1)
        X = np.append(X, one_hot_date, axis=1)
        return (X, y)

    X, y = load_file("fullDeliverable.db")
    x_train, x_test, y_train, y_test = train_test_split(X, y, p)
    X = sm.add_constant(x_train)
    model = sm.OLS(y_train, X) 
    results = model.fit() 
    print(results.summary())
    
    average = np.mean(y_train)
    train_predict = results.predict(X)
    train_MSE = eval_measures.mse(y_train, train_predict)
    regular_train_MSE = eval_measures.mse(y_train, average)
    regular_test_MSE = eval_measures.mse(y_test, average)
    X1 = sm.add_constant(x_test)
    test_predict = results.predict(X1)
    test_MSE = eval_measures.mse(y_test, test_predict)

    print("Baseline Train MSE: " + str(regular_train_MSE))
    print("Baseline Test MSE: " + str(regular_test_MSE))
    print("Regression Train MSE: " + str(train_MSE))
    print("Regression Test MSE: " + str(test_MSE))


    