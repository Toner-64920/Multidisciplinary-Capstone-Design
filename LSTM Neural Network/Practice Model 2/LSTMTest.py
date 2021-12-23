'''
This LTSM IS multivariable based, but has some features that make it so that it can be used as a reference
for a multivariable approach in the future.

This simple LSTM is based around forecasting monthly production of milk. There are comments on the vast majority of functions
in the code. THIS IS DIFFERENT FROM THE INITIAL PRACTICE LSTM MODEL. Due to how this is based on multiple variables and how, after
way too many hours of research, it uses different methods of plotting the forecast model and making predictions, this is different
in some pretty big ways from the original practice LSTM. It uses the same data, so you don't need to download another csv, since the
other variables/features used are gathered from the seasonal decompose of the data in the code. This is just meant to get ideas for
multiple variables in LSTM (this implimentation isn't the best).
'''

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wid

from pathlib import Path
from matplotlib import dates
#Not needed for this implimentation, I think
from keras.preprocessing.sequence import TimeseriesGenerator

def combine_seasonal_cols(input_df, results):
    #Adds new seasonal cols to df given the seasonal results
    #Input df is a data frame, while results is a statesmodels decompose result
    
    input_df["observed"] = results.observed
    input_df["residual"] = results.resid
    input_df["seasonal"] = results.seasonal
    input_df["trend"] = results.trend

#Prevents Pandas from printing error about pd (for test values, false positives)
pd.options.mode.chained_assignment = None

'''
Reads data from monthly milk production csv and puts it into a data frame df.
Put the monthly_milk_production.csv file in the same folder as where ever this .py file is located on your computer
'''
CURR_PATH = Path(__file__)
DATA_PATH = CURR_PATH.parents[1]
df = pd.read_csv(os.path.join(DATA_PATH, "monthly_milk_production.csv"), index_col = "Date", parse_dates = True)
#For testing, if OS method doesn't work for some reason, uncomment this and put where the data csv file is located after the r
#df = pd.read_csv(r'Whereever\On-Your-Computer\--->.csv\is-located\monthly_milk_production.csv',index_col='Date',parse_dates=True)

#Specifying that this is monthly based data
df.index.freq = "MS"

df.head()

#Plots df over the 12 years that there is data in milk production for
df.plot(figsize = (12,6))

#This sets the Y axis label (there wasn't one by default)
plt.ylabel("Milk Produced")
plt.title("Milk Production Data")

#Gets and plots seasonal stat models, then gives info on trends, seasonality, and residuality through plots
from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(df["Production"])
results.plot();
plt.xlabel("Year")
#Adds seasonal decompose data to the milk production data as separate columns
combine_seasonal_cols(df, results)
#Drops any NaN's from df after adding the seasonal decompose data
df.dropna(axis = 1, inplace = True)
#For testing
#print(df)

#Sets training and testing variables (training data is up to month 156, test data is the last 12 months, there are 168 in total)
train = df.iloc[:156]
test = df.iloc[156:]
#Splits up data (y is production, x is everything else)
x_train, y_train = train.drop(train.iloc[:, 0:1], axis = 1), train.iloc[:, 0:1]
x_test, y_test = test.drop(train.iloc[:, 0:1], axis = 1), test.iloc[:, 0:1]

#For Testing
#print(train)
#print(test)


#Scaling (scales values to work with the LSTM model method, values should be between 0 and 1 so that the range isn't too big)
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
#scaler_y = MinMaxScaler()

#For testing
df.head(),df.tail()

scaler_x.fit(x_train)
#scaler_y.fit(y_train)

#This will scale the values in train and test from 0 to 1
scaled_x_train = scaler_x.transform(x_train)
scaled_x_test = scaler_x.transform(x_test)
#Reshaping data so that it works properly when forecasting
x_train_reshaped = scaled_x_train.reshape((scaled_x_train.shape[0], 1, scaled_x_train.shape[1]))
x_test_reshaped = scaled_x_test.reshape((scaled_x_test.shape[0], 1, scaled_x_test.shape[1]))
#Not used in this model, but here if needed in the future, scales y values (production)
#scaled_y_train = scaler_y.transform(y_train)
#scaled_y_test = scaler_y.transform(y_test)

#For testing
print(y_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Number of months being forecasted/predicted in this model
pred_nums = 12
#Number of columns in the x_train and x_test dfs, there are 2 in this model, since milk production is in the y lists, and the other seasonal data had too many NaN values so they were removed
n_features = 2

# defines the model (since this is a LSTM test the model is an LSTM model)
model = Sequential()
#This sets the model to an LSTM model with 100 neurons per input layer, with an input shape of (12, 2), 12 for the 12 months being predicted, 2 for the 2 input variables from the x arrays
model.add(LSTM(100, activation = "relu", input_shape = (pred_nums, n_features)))
model.add(Dense(1))

#This just sets the optimizer to adam and losses to be based around mean squared error (adam is being used since it is based around adaptive estimation and learning, making it a good optimizer for the model)
model.compile(optimizer = "adam", loss = "mse")

#For testing
model.summary()

#Fits model over 70 epochs (this is the training portion of the Neural network, doesn't use a time series generator anymore)
history = model.fit(x_train_reshaped, y_train, validation_data = (x_test_reshaped, y_test), epochs = 20, batch_size = 12, verbose = 2, shuffle = False)
print(history)
y_pre = model.predict(x_test_reshaped)
print(y_pre)

#This just shows the loss that is happening over every epoch through a graph, good way to see if it is learning/reducing error with each epoch
loss_per_epoch = history.history["loss"]
#Creates figure for this graph to be plotted on
fig, ax1 = plt.subplots(1, 1)
ax1.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.ylabel("Value Loss Per Epoch")
plt.xlabel("Number of Epochs (Completed)")
plt.title("Value Loss Per Epoch")

#Adds the prediction values to the y_test data frame so that the predictions and actual 12 months of data are on the same plot properly (the predictions aren't way earlier on the graph than the actual values)
test_predictions = []

for i in range(len(y_pre)):
    
    # get the prediction array at i
    current_array = y_pre[i]
    
    #Get the prediction value from the array (it is always at 0)
    current_pred = current_array[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 

#Add the predictions as a new column in the y_test data frame
y_test["Forecast"] = test_predictions
print(y_test)

#Plots the model prediction with the actual data
fig, (ax, axx) = plt.subplots(2, 1)

ax.plot(y_test["Production"])
ax.set_title("Actual Data", fontsize = 8)
plt.xticks()
ax.set_ylabel("Milk Produced")

axx.plot(y_test["Forecast"])
axx.set_title("Milk Production Forecast (Predictions)", fontsize = 8)
plt.xlabel("Year-Month")
plt.xticks()
axx.set_ylabel("Milk Produced")

# Adjust layout to make room for the table:
plt.subplots_adjust(left = 0.1, bottom = 0.086, top = 0.96)

#Calculates root mean squared error of the prediction model and then prints it
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test["Production"], y_test["Forecast"]))
print("Root Mean Squared Error: " + str(rmse))

#Percent error plot
percent_error = []
for i in range(len(test_predictions)):
    test_curr = y_test.iloc[i, 0]
    error_current = abs(((test_predictions[i] - test_curr)/test_curr) * 100)
    percent_error.append(error_current)

print(percent_error)

fig, axxx = plt.subplots(1, 1)
axxx.plot(range(len(percent_error)), percent_error)
axxx.set_ylabel("Percent error (%)")
axxx.set_xlabel("Month (in 1975)")
axxx.set_title("Percent error in 1975 (Prediction year)")
