'''
This is just meant to create a csv with the data for the final LSTM model. The data used
will be based around specific item categories and state in the final model. Due to how much
time this takes each time the LSTM model runs, this is now being done in a separate file (this one)
and the csv will be used in the model. Essentially this is just being done to reduce the actual model run times

This code is very very messy and not up to normal standards. Read documentation for more details.
'''
import os
import time
import pandas as pd
import numpy as np
import math
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
    
#Returns time passed as hours:minutes:seconds from start time to when this function is called, need time module
def get_time_elapsed(start_time):
    #Start time should idealy be when the program starts running, but it can be another value if desired
    e = int(time.time() - start_time)
    print("Time elapsed: {:02d}:{:02d}:{:02d}".format(e // 3600, (e % 3600 // 60), e % 60))    

def add(a, b):
    return a + b

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

if __name__ == "__main__":
    start_time = time.time()
    CURR_PATH = Path(__file__)
    DATA_PATH = CURR_PATH.parents[1]
    SALES_TRAIN_VAL_PATH = os.path.join(DATA_PATH, "Data/sales_train_validation.csv")
    SALES_TRAIN_EVAL_PATH = os.path.join(DATA_PATH, "Data/sales_train_evaluation.csv")
    CALENDAR_PATH = os.path.join(DATA_PATH, "Data/calendar.csv")  
    RAINFALL_PATH = os.path.join(DATA_PATH, "Data/rainfall.csv")
    SALES_MOD_PATH = os.path.join(DATA_PATH, "Data/sales_train_evaluation_summed_by_day.csv")
    print("All OS paths have been set")
    get_time_elapsed(start_time)
    
    #This creates the dfs used
    cal_data = pd.read_csv(CALENDAR_PATH, index_col = "date", parse_dates = True)
    train_val = pd.read_csv(SALES_TRAIN_VAL_PATH)
    train_eval = pd.read_csv(SALES_TRAIN_EVAL_PATH)
    rainfall_data = pd.read_csv(RAINFALL_PATH)
    mod_sales_data = pd.read_csv(SALES_MOD_PATH)
    OUTPUT = "sales_train_mod.csv"
    print("All files found and read")
    get_time_elapsed(start_time)
    '''
    #Features will be events, rainfall, and sale price averages (add seasonality after actual sales with decompose)
    neural_data_frame = cal_data.copy()
    #Drops weekday and d, wday will be used to determine day of the week with day 1 being saturday and day 7 being friday
    neural_data_frame.drop(columns = "weekday", axis = 1, inplace = True)
    neural_data_frame.drop(columns = "d", axis = 1, inplace = True)
    
    #Rainfall stuff
    neural_data_frame["Rainfall_CA"] = rainfall_data["CA"].tolist()
    neural_data_frame["Rainfall_TX"] = rainfall_data["TX"].tolist()
    neural_data_frame["Rainfall_WI"] = rainfall_data["WI"].tolist()
	
    neural_data_frame = neural_data_frame.iloc[:1913]
    print(len(neural_data_frame))
    #Actual sales, this is the y data frame (don't add to neural df)
    train_val_formatted = format_data(train_val, cal_data)
    all_item_ids = train_eval["item_id"].tolist()
    print(all_item_ids)
    
    #To show it is working
    print("The neural network data frame has been created (Needs Seasonal decompose)")
    get_time_elapsed(start_time)
    print(neural_data_frame.columns)
    print(neural_data_frame)
    '''
    sales_mod_validation = mod_sales_data.iloc[:, 1919:].copy()
    sales_mod_validation = sales_mod_validation.dropna(axis = 1)
    get_time_elapsed(start_time)
    print(sales_mod_validation)
    jimbo = mod_sales_data.iloc[:, 0:6].copy()
    print(jimbo)
    jimbo = jimbo.join(sales_mod_validation)
    jimbo = jimbo.dropna(axis = 1)
    print(jimbo)
    
    #This part just gets the sales from the modified sheet and adds them
    #Hobbies Sales
    #Rows 2, 3, 9, 10, 16, 17, 23, 24, 30, 31, 37, 38, 44, 45, 51, 52, 58, 59, 65, 66
    '''
    print(mod_sales_data)
    hobbies_sales = mod_sales_data.loc[2-2].iloc[6:1919].tolist()
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[3-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[9-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[10-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[16-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[17-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[23-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[24-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[30-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[31-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[37-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[38-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[44-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[45-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[51-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[52-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[58-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[59-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[65-2].iloc[6:1919].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (mod_sales_data.loc[66-2].iloc[6:1919].tolist())))
    hobbies_sales = [item for item in hobbies_sales if not(math.isnan(item)) == True]
    print(hobbies_sales)
    neural_data_frame["Hobbies_Sales"] = hobbies_sales
    '''
    hobbies_sales = sales_mod_validation.loc[2-2].tolist()
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[3-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[9-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[10-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[16-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[17-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[23-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[24-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[30-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[31-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[37-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[38-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[44-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[45-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[51-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[52-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[58-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[59-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[65-2].tolist())))
    hobbies_sales = list(map(add, hobbies_sales, (sales_mod_validation.loc[66-2].tolist())))
    hobbies_sales = [item for item in hobbies_sales if not(math.isnan(item)) == True]
    print(hobbies_sales)
    jimbo2 = pd.DataFrame({"Hobbies_Sales_Total": hobbies_sales})
    print(jimbo2)
    
    #Foods Sales
    #6 7 8 13 14 15 20 21 22 27 28 29 34 35 36 41 42 43 48 49 50 55 56 57 62 63 64 69 70 71?
    '''
    food_sales = mod_sales_data.loc[6-2].iloc[6:1919].tolist()
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[7-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[8-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[13-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[14-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[15-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[20-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[21-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[22-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[27-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[28-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[29-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[34-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[35-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[36-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[41-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[42-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[43-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[48-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[49-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[50-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[55-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[56-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[57-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[62-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[63-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[64-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[69-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[70-2].iloc[6:1919].tolist())))
    food_sales = list(map(add, food_sales, (mod_sales_data.loc[71-2].iloc[6:1919].tolist())))
    food_sales = [item for item in food_sales if not(math.isnan(item)) == True]
    print(food_sales)
    neural_data_frame["Food_Sales"] = food_sales
    '''
    food_sales = sales_mod_validation.loc[6-2].tolist()
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[7-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[8-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[13-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[14-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[15-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[20-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[21-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[22-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[27-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[28-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[29-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[34-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[35-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[36-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[41-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[42-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[43-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[48-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[49-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[50-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[55-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[56-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[57-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[62-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[63-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[64-2].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[67].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[68].tolist())))
    food_sales = list(map(add, food_sales, (sales_mod_validation.loc[69].tolist())))
    food_sales = [item for item in food_sales if not(math.isnan(item)) == True]
    print(food_sales)
    jimbo2["Food_Sales_Total"] = food_sales
    
    #Household Sales
    #4 5 11 12 18 19 25 26 32 33 39 40 46 47 53 54 60 61 67 68 
    '''
    household_sales = mod_sales_data.loc[4-2].iloc[6:1919].tolist()
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[5-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[11-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[12-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[18-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[19-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[25-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[26-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[32-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[33-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[39-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[40-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[46-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[47-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[53-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[54-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[60-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[61-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[67-2].iloc[6:1919].tolist())))
    household_sales = list(map(add, household_sales, (mod_sales_data.loc[68-2].iloc[6:1919].tolist())))
    household_sales = [item for item in household_sales if not(math.isnan(item)) == True]
    neural_data_frame["Household_Sales"] = household_sales
    '''
    household_sales = sales_mod_validation.loc[4-2].tolist()
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[5-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[11-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[12-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[18-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[19-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[25-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[26-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[32-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[33-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[39-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[40-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[46-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[47-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[53-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[54-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[60-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[61-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[67-2].tolist())))
    household_sales = list(map(add, household_sales, (sales_mod_validation.loc[68-2].tolist())))
    household_sales = [item for item in household_sales if not(math.isnan(item)) == True]
    jimbo2["Household_Sales_Total"] = household_sales
    print(jimbo2.columns)
    print(jimbo2)
    
    #CA Sales
    #CA_sales = mod_sales_data.loc[0].iloc[6:1919].tolist()
    CA_sales_2 = sales_mod_validation.loc[0].tolist()
    for i in range(1, 29):
        #CA_sales = list(map(add, CA_sales, (mod_sales_data.loc[i].iloc[6:1919].tolist())))
        CA_sales_2 = list(map(add, CA_sales_2, (sales_mod_validation.loc[i].tolist())))
    #neural_data_frame["CA_Sales"] = CA_sales
    CA_sales_2 = [item for item in CA_sales_2 if not(math.isnan(item)) == True]
    print(CA_sales_2)
    jimbo2["CA_Sales_Total"] = CA_sales_2
    
    #TX Sales
    #TX_sales = mod_sales_data.loc[29].iloc[6:1919].tolist()
    TX_sales_2 = sales_mod_validation.loc[29].tolist()
    for i in range(30, 50):
        #TX_sales = list(map(add, TX_sales, (mod_sales_data.loc[i].iloc[6:1919].tolist())))
        TX_sales_2 = list(map(add, TX_sales_2, (sales_mod_validation.loc[i].tolist())))
    #neural_data_frame["TX_Sales"] = TX_sales
    TX_sales_2 = [item for item in TX_sales_2 if not(math.isnan(item)) == True]
    jimbo2["TX_Sales_Total"] = TX_sales_2
    
    #WI Sales
    #WI_sales = mod_sales_data.loc[50].iloc[6:1919].tolist()
    WI_sales_2 = sales_mod_validation.loc[50].tolist()
    for i in range(51, (mod_sales_data.count()[0] - 1)):
        #WI_sales = list(map(add, WI_sales, (mod_sales_data.loc[i].iloc[6:1919].tolist())))
        WI_sales_2 = list(map(add, WI_sales_2, (sales_mod_validation.loc[i].tolist())))
    #neural_data_frame["WI_Sales"] = WI_sales
    WI_sales_2 = [item for item in WI_sales_2 if not(math.isnan(item)) == True]
    jimbo2["WI_Sales_Total"] = WI_sales_2
    
    print(jimbo2.columns)
    print(jimbo2)
    get_time_elapsed(start_time)
    #neural_data_frame.to_csv(OUTPUT, index = True)
    jimbo2.to_csv("validation_mod.csv", index = True)