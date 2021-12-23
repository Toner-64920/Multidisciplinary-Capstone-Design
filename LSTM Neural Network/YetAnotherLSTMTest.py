'''
This LSTM is a test with the actual data from the competition. This combines the methods and techniques used
in LSTMTest.py and LSTMTest_OLD.py, as well as other methods either found from research or learned from other
group members. Methods that are carried over or already have information on what it is doing on the EDN forum
post for the LSTM implementation or in the other files themselves won't have as much information in this files
comments on them in order to not have way too many comments in the file.

Main differences between this and multivariate practice model:
-The main focus, the overal sales for all of these walmart locations, will be over a bigger range
(last 28 days of sales versus the last 12 months of milk production, 28 data points instead of 12)

*-There are 3 y values being predicted rather than just 1, the 3 sales forecasts for each store (this will
change the shaping of the data and the final plot displyed at the end)

-There is much more supplemental data to help train the neural network now that rainfall,
yearly events, and [the categories of items being sold]* are now given to be used in the training process

-This program takes much longer to run (when you hit the run button be prepared for it to take at least 1 min
unless you secretly have a good I9 processor and >16 gb of ram, also ram usage jumping around in the 2,000 mb to 6,780 mb
is normal, don't panic if this happens)

-Since there are 3 forecasts being done now, there will be much more information on the final forecast plot
being shown after the program is done running. As a result of this, the final forecast plot will have more
information on it in order to make the final output more clear (planned to have buttons to switch to certain
plots and a page dedicated solely for certain stats to give a better idea of how the model performed, how in
depth this is will depend on how much time is left between when the initial model is done and when the project is due)
'''

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wid
import tensorflow as tf

from pathlib import Path
from matplotlib import dates
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def combine_seasonal_cols(input_df, results):
    input_df["residual"] = results.resid
    input_df["seasonal"] = results.seasonal
    input_df["trend"] = results.trend

def get_sales_dates(df, cal_df, interval=1):
	
	# columns: 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'
	
	sales = df[df.columns[6:]]
	num_days = len(sales.columns)
	cal_data = cal_df.copy(deep=True)[:num_days]
	sales_dates = cal_data.loc[cal_data["d"] == sales.columns, "date"].unique()
	sales_dates = dates.datestr2num(sales_dates)

	return sales_dates[::interval]

def plot_col_unique(df, cal_df, col, stat = "mean", interval = 1):
	'''
	Plot sales for each unique value of a column in
	the sales data. Statistic must be an aggregate
	(mean, median, etc.)
	'''
	num_cols = len(df.columns[6:])
	idxs_interval = np.arange(num_cols) // interval
	sales_interval = df[df.columns[6:]].groupby(idxs_interval, axis = 1).mean()
	df_interval = pd.concat([df[df.columns[:6]], sales_interval], axis = 1)

	sales_group = df_interval.groupby([col])[df_interval.columns[6:]]
	groups = [group[0] for group in sales_group]

	# apply statistic
	if stat == "mean":
		sales_group = sales_group.mean()
	elif stat == "median":
		sales_group = sales_group.median()
	elif stat == "mode":
		sales_group = sales_group.mode()
	elif stat == "max":
		sales_group = sales_group.max()
	elif stat == "min":
		sales_group = sales_group.min()
	elif stat == "sum":
		sales_group = sales_group.sum()
	else:
		assert(False, "Statistic unknown/not implemented")

	sales_dates = get_sales_dates(df, cal_df, interval)
	sales_np = sales_group.to_numpy().T

	plt.plot(sales_dates, sales_np)
	plt.legend(groups, loc = 'best')

def format_data(df, cal_df):
	'''
	One row contains:
		- Product ID
		- Product Info (product category, state, store ID, etc.)
		- Unit sales for particular day
		- Day
	'''
	products_df = df[df.columns[:6]]
	sales_df = df[df.columns[6:]]
	num_products = len(products_df)

	# duplicate product rows for each day
	num_days = len(sales_df.columns)
	products_df = products_df.append([products_df] * (num_days - 1), ignore_index=True)
	products_df = products_df.sort_values(by = ["id"], ignore_index = True)

	sales_df = sales_df.T
	sales_np = np.concatenate([sales_df[col].values for col in sales_df.columns])
	products_df["num_sales"] = sales_np

	return products_df

#Returns time passed as hours:minutes:seconds from start time to when this function is called, need time module
def get_time_elapsed(start_time):
    #Start time should idealy be when the program starts running, but it can be another value if desired
    e = int(time.time() - start_time)
    print("Time elapsed: {:02d}:{:02d}:{:02d}".format(e // 3600, (e % 3600 // 60), e % 60))    

if __name__ == "__main__":
    #Look up how to store data on LSTM Implementation EDN post
    start_time = time.time()
    CURR_PATH = Path(__file__)
    DATA_PATH = CURR_PATH.parents[1]
    SALES_TRAIN_EVAL_PATH = os.path.join(DATA_PATH, "Data/validation_mod.csv")
    NEURAL_PATH = os.path.join(DATA_PATH, "Data/sales_train_mod.csv")
    print("All OS paths have been set")
    get_time_elapsed(start_time)
    
    #This creates the dfs used
    train_eval_frame = pd.read_csv(SALES_TRAIN_EVAL_PATH, index_col = "Date", parse_dates = True)
    neural_data_frame = pd.read_csv(NEURAL_PATH, index_col = "date", parse_dates = True)
    print("All files found and read")
    get_time_elapsed(start_time)
    #This is just the name of the column being used for the y values in the model
    #Later on replace all of the instances where this is typed out with this variable as well as the column number
    forecast_category = "Food_Sales"
    #Make sure that the forecast category is typed correctly, or else this won't work
    forecast_col_num = neural_data_frame.columns.get_loc(forecast_category)
    print("Forecast category: " + forecast_category + " located in column " + str(forecast_col_num))
    
    fig, axxx = plt.subplots(1, 1)
    axxx.plot(neural_data_frame["Food_Sales"])
    axxx.set_ylabel("Number of sales (Food)")
    axxx.set_xlabel("Date")
    axxx.set_title("Forecast data plot (not predictions)")
    
    #Seasonal decompose
    from statsmodels.tsa.seasonal import seasonal_decompose
    results = seasonal_decompose(neural_data_frame["Food_Sales"])
    results.plot()
    plt.xlabel("Year")
    plt.title("Seasonal Decompose")
    
    #Adds seasonal decompose data to the unit sales data as separate columns
    combine_seasonal_cols(neural_data_frame, results)
    #Drops any NaN's from df after adding the seasonal decompose data
    neural_data_frame.dropna(axis = 1, inplace = True)
    print(neural_data_frame.columns)
    
    print("The neural network data frame has been created")
    get_time_elapsed(start_time)
    #Sets training and testing variables, x is everythin EXCEPT actual sales
    train = neural_data_frame.iloc[:(len(neural_data_frame) - 28)]
    print(train.columns)
    test = neural_data_frame.iloc[(len(neural_data_frame) - 28):]
    print(test.columns)
    #Splits up data (y is production, x is everything else)
    x_train, y_train = train.drop(train.iloc[:, 10:16], axis = 1), train.iloc[:, 11:12]
    x_test, y_test = test.drop(train.iloc[:, 10:16], axis = 1), test.iloc[:, 11:12]
    
    #For Testing
    print("The training data has been created")
    get_time_elapsed(start_time)
    
    #Scaling (scales values to work with the LSTM model method, values should be between 0 and 1 so that the range isn't too big)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_x.fit(x_train)
    scaler_y.fit(y_train)
    
    #This will scale the values in train and test from 0 to 1
    scaled_x_train = scaler_x.transform(x_train)
    scaled_x_test = scaler_x.transform(x_test)
    print(scaled_x_train)
    print(scaled_x_test)
    #Reshaping data so that it works properly when forecasting
    x_train_reshaped = scaled_x_train.reshape((scaled_x_train.shape[0], 1, scaled_x_train.shape[1]))
    x_test_reshaped = scaled_x_test.reshape((scaled_x_test.shape[0], 1, scaled_x_test.shape[1]))
    
    #For testing
    print("The training data has been scaled")
    get_time_elapsed(start_time)
    print("Scaled X training data:")
    print(scaled_x_train.shape)
    print("Scaled X test data:")
    print(scaled_x_test.shape)

    #Number of months being forecasted/predicted in this model
    pred_nums = 28
    #Number of columns in the x_train and x_test dfs, there are 2 in this model, since milk production is in the y lists, and the other seasonal data had too many NaN values so they were removed
    #For future groups, change this from 11 to len(scaled_x_train.columns) or len(x_train.columns)
    n_features = 11
    
    # defines the model (since this is a LSTM test the model is an LSTM model)
    model = Sequential()
    #This sets the model to an LSTM model with 100 neurons per input layer, with an input shape of (12, 2), 12 for the 12 months being predicted, 2 for the 2 input variables from the x arrays
    model.add(LSTM(100, activation = "relu", input_shape = (1, n_features)))
    #For future groups, change layers so that there are more
    model.add(Dense(1))
    
    #This just sets the optimizer to adam and losses to be based around mean squared error (adam is being used since it is based around adaptive estimation and learning, making it a good optimizer for the model)
    model.compile(optimizer = "adam", loss = "mse")
    
    #For testing
    model.summary()
    print("The LSTM model and conditions have been created")
    get_time_elapsed(start_time)
    
    #Fits model over 70 epochs (this is the training portion of the Neural network, doesn't use a time series generator anymore)
    history = model.fit(x_train_reshaped, y_train, validation_data = (x_test_reshaped, y_test), epochs = 2500, batch_size = 12, verbose = 2, shuffle = False)
    print(history)
    y_pre = model.predict(x_test_reshaped)
    print(y_pre)
    
    '''
    #This just shows the loss that is happening over every epoch through a graph, good way to see if it is learning/reducing error with each epoch
    loss_per_epoch = history.history["loss"]
    #Creates figure for this graph to be plotted on
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(range(len(loss_per_epoch)), loss_per_epoch)
    plt.ylabel("Value Loss Per Epoch")
    plt.xlabel("Number of Epochs (Completed)")
    plt.title("Value Loss Per Epoch")
    '''
    
    #Adds the prediction values to the y_test data frame so that the predictions and actual 12 months of data are on the same plot properly (the predictions aren't way earlier on the graph than the actual values)
    test_predictions = []
    
    for i in range(len(y_pre)):
	
	# get the prediction array at i
        current_array = y_pre[i]
	
	#Get the prediction value from the array (it is always at 0)
        current_pred = current_array[0]
	
	# append the prediction into the array
        test_predictions.append(current_pred) 
    
    #Add the predictions as a new column in the y_test data frame, errors start here
    final_data_frame = neural_data_frame.iloc[:, 11:12].copy()
    final_data_frame = final_data_frame.iloc[len(final_data_frame) - 28:]
    final_data_frame["Forecast"] = test_predictions
    print(final_data_frame)
    
    #Plots the model prediction with the actual data
    fig, (ax, axx) = plt.subplots(2, 1)
    
    ax.plot(final_data_frame["Food_Sales"])
    #ax.plot(y_pre, label = "Predicted")
    ax.set_title("Actual Data", fontsize = 8)
    plt.xticks()
    ax.set_ylabel("Number of sales")
    
    axx.plot(final_data_frame["Forecast"])
    axx.set_title("Sales Forecast", fontsize = 8)
    plt.xlabel("Year-Month-Date")
    plt.xticks()
    axx.set_ylabel("Number of sales")
    
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left = 0.1, bottom = 0.086, top = 0.96)
    
    #Calculates root mean squared error of the prediction model and then prints it
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rmse = sqrt(mean_squared_error(final_data_frame["Food_Sales"], final_data_frame["Forecast"]))
    print("Root Mean Squared Error: " + str(rmse))
    
    #Percent error plot
    percent_error = []
    for i in range(len(test_predictions)):
        test_curr = final_data_frame.iloc[i, 0]
        if test_curr == 0:
            error_current = 100
        else:
            error_current = abs(((test_predictions[i] - test_curr)/test_curr) * 100)
        percent_error.append(error_current)
    
    print(percent_error)
    
    fig, axxx = plt.subplots(1, 1)
    axxx.plot(range(len(percent_error)), percent_error)
    axxx.set_ylabel("Percent error (%)")
    axxx.set_xlabel("Day")
    axxx.set_title("Percent error (Predictions)")
    #Add feature that makes it so that the plot color changes with the error (80+ Red, 60-80 orange, 40-60 yellow, 20-40 limish yellow, 0-20 green)?
    #Add table at the bottom showing values (data, then predictions, then percent error)
    print("The forecast and percent error have been found and displayed")
    print("Total run time:")
    get_time_elapsed(start_time)