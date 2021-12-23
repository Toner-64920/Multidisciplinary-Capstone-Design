'''
This LTSM ISN'T multivariable based, but has some features that make it so that it can be used as a reference
for a multivariable approach in the future.

This simple LSTM is based around forecasting monthly production of milk. There are comments on the vast majority of functions
in the code.
'''

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib import dates

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

#Gets and plots seasonal stat models, then gives info on trends, seasonality, and residuality through plots
from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(df["Production"])
results.plot();

#Sets training and testing variables (training data is up to month 156, test data is the last 12 months, there are 168 in total)
train = df.iloc[:156]
test = df.iloc[156:]

#Scaling (scales values to work with the LSTM model method, values should be between 0 and 1 so that the range isn't too big)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#For testing
#df.head(),df.tail()

scaler.fit(train)
#This will scale the values in train and test from 0 to 1
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

#For testing
#scaled_train[:10]

from keras.preprocessing.sequence import TimeseriesGenerator


# define the times series generator with the training data (this one is for 3 months)
n_input = 3
#This is what needs to change if there are more time series, for this LSTM there is only one, but would change this if there were >1
n_features = 1
#Generator in this model predicts a value from three given values (n_input)
generator = TimeseriesGenerator(scaled_train, scaled_train, length = n_input, batch_size = 1)
X,y = generator[0]
print(f"Given the Array: \n{X.flatten()}")
print(f"Predict this y: \n {y}")
'''
Essentially the generator gets a number of inputs, and then predicts the next value based off these inputs, the given array
and predict this y print statments are just showing that
'''
X.shape

# Now this is for 12 months (12 input values)
n_input = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length = n_input, batch_size = 1)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# defines the model (since this is a LSTM test the model is an LSTM model)
model = Sequential()
#This sets the model to an LSTM model with 100 neurons
model.add(LSTM(100, activation = "relu", input_shape = (n_input, n_features)))
model.add(Dense(1))
#This just sets the optimizer to adam and losses to be based around mean squared error (adam is being used since it is based around adaptive estimation and learning, making it a good optimizer for the model)
model.compile(optimizer = "adam", loss = "mse")

#For testing
model.summary()

#fits model over 50 epochs (this is the training portion of the Neural network)
model.fit(generator, epochs = 50)

#This just shows the loss that is happening over every epoch through a graph, good way to see if it is learning/reducing error with each epoch
loss_per_epoch = model.history.history["loss"]
#Creates figure for this graph to be plotted on
fig, ax1 = plt.subplots(1, 1)
ax1.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.ylabel("Loss Per Epoch")
plt.xlabel("Number of Epochs")
plt.title("Loss' Per Epoch")

#Takes last 12 month values in the training data set to make a prediction for the first value in the test set
last_train_batch = scaled_train[-12:]
#Reshapes data so that it is in a similar format to the data the model was trained with (same as X.shape above)
last_train_batch = last_train_batch.reshape((1, n_input, n_features))
model.predict(last_train_batch)
scaled_test[0]


test_predictions = []

#Takes last 12 values in the training set
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis = 1)

test.head()

#This takes the test predictions and rescales them back to their original values (they are still just from 0-1 before this)
true_predictions = scaler.inverse_transform(test_predictions)
test["Predictions"] = true_predictions

#Plots the model prediction with the actual data
test.plot(figsize = (14, 6))
plt.ylabel("Milk Produced")

#Calculates root mean squared error of the prediction model and then prints it
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test["Production"], test["Predictions"]))
print(rmse)