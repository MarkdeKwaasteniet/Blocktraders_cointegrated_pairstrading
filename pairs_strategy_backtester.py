import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime
import h5py
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar import vecm
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def df_creator(name):                                                                   #Create an orderbook dataframe for every month/seperate h5 file.
    file = "lib/orderbook_" + name + ".h5"                                              #Read in the files example for mach: "lib/orderbook_march.h5"  
    data_file = h5py.File(file,'r')                                                     #Use h5py package to read in the h5 file.
    df_total = pd.DataFrame(data = np.array(data_file['/data/block0_values']))          #Create a DataFrame containing all the values in the orderbook
    df_total.columns = np.array(data_file['/data/block0_items'])                        #Fill in the column names using the block0_items: Example: b'bidsP:0|binance-perps|XRP-USDT_PS'
    return df_total                                                                     #Return dataframe

def datetime_index(start_date, frequency, periods):                                     #Simple function that returns the 0 --> 432000 seconds into actual datetime values dependent on the begin date
    start_dt = datetime.fromisoformat(start_date)
    dt_range = pd.date_range(start_dt, periods=periods, freq=frequency)
    return dt_range

def sample_processor(df, days, frequency, begin_date):                                  #Simple function that sets the size of the sample, the frequency and is used to seperate the various months
    total_seconds = (60*60*24*days)
    df_days = df.iloc[:total_seconds,:]
    df_days.index = datetime_index((begin_date + " 00:00:00"), frequency, len(df_days))
    return df_days

def deep_price(df, column_P, column_Q,split_point,coins):                                                           #A function to calculate the quantity weighted average mid-point price
    df_coins = {}                                                                                                   #Create dictionary to store the price of every coin seperately
    for i in range(0,int(len(coins))):                                                                              #Loop over the number of coins in the dataset: in our example this is 6
        df_order_book = df.iloc[:,((split_point*2)*i):((split_point*2)+(split_point*2)*i)]                          #Create a df of 40 columns every loop, which contain bid prices, bid quantities, ask prices, ask quantities
        
        thresh_Q = list(coins.values())[i]                                                                          #Set the quantity threshold for a certain coin. Based on its name index in the dictionary "coins"                                                                          
                                                                                                                    #XRP is the first coin, so index is 0. Resulting in a quantity of 4500 XRP --> approx 2000 USD.
        def price_calc(df_split):
            price = []
            for index, row in df_split.iterrows():                                                                  #Iterate over the splitted dataframe which contains only bids or asks (quantity and prices)
                if df_split.iloc[index,column_Q] >= thresh_Q:                                                       #If quantity is above the determined threshold, take the first bid/ask price
                    true_price = df_split.iloc[index,column_P] * thresh_Q                                           #price will be multiplied by threshold, at the end of the loop it will be divided by same quantity.
                
                else:
                    quantity = thresh_Q                                                                             #If quantity is not above the determine threshold, create while loop in which it will calculate weighted price
                    book_dept = 0                                                                                   #Book_dept indicates the number of steps the prices has gone into the orderbook.
                    true_price = 0                                                                                  #True price is considered the quantity weighted price
                    
                    while quantity > 0:                                                                             #While the quantity is not yet equal to the threshold, go deeper into orderbook
                        if quantity > df_split.iloc[index,book_dept+column_Q]:                                      #If quantity available is smaller than (still) required quantity for threshold:
                            true_price += (df_split.iloc[index,book_dept]*df_split.iloc[index,book_dept+column_Q])  #Multiply price by the available quantity
                            quantity -= df_split.iloc[index,book_dept+column_Q]                                     #Reduce the required quantity by the available quantity that is bought
                            
                            if book_dept < 9:                                                                       #If the end of the orderbook is not yet met.
                                book_dept += 1                                                                      #Take another step into the orderbook
                            else:
                                true_price += df_split.iloc[index,book_dept]*quantity                               #If the end of the orderbook is met, use the last quantity/price to fill the rest of the required quantity.
                                quantity -= df_split.iloc[index,book_dept+column_Q]                                 #Reduce the required quantity by the available quantity that is bought, which in this case is all.
                                break                                                                               #Break the fore loop en determine next price
                            
                        else:
                            true_price += df_split.iloc[index,book_dept]*quantity                                   #If the quantity available is larger than the still required quantity for threshold, buy rest of quantity needed at this price
                            break                                                                                   #Break the fore loop en determine next price   
                
                weighted_price = true_price / thresh_Q                                                              #prices have been multiplied by threshold quantity, to get the actual price back, divide it by Threshold_Q
                price.append(weighted_price)                                                                        #Append the price to the list of prices
            return price
        
        bid_price = price_calc(df_order_book.iloc[:,:split_point])                                                  #Return the deep prices in the form of a list
        ask_price = price_calc(df_order_book.iloc[:,split_point:])
        
        df_coins[list(coins.keys())[i]] =  (np.array(bid_price) + np.array(ask_price)) / 2                          #Calculate the weighted average mid point, by taking the averages of the bid and ask deep prices

    
    return pd.DataFrame.from_dict(df_coins)                                                                         #Return the dictionary of all the coins which include the prices

def dataframe_scaler(df_input, index, n_obs, scaler):                                               # Simple function to scale the dataframe using the input of the user.
    df = df_input.iloc[index*n_obs:n_obs+index*n_obs,:]     
    
    if scaler == "standardize":                                                                     # Based on the argument scale the dataframe on that method
        df_standard = df.copy()
        df_standard[df_standard.columns] = StandardScaler().fit_transform(df_standard[df_standard.columns])
        return df_standard, df
    
    elif scaler == "normalize":
        df_norm = df.copy()
        for c in df_norm.columns:
            df_norm[c] = df_norm[c]/np.linalg.norm(df_norm[c])
        return df_norm, df

    elif scaler == "minmax":
        df_minmax = df.copy()
        df_minmax[df_minmax.columns] = MinMaxScaler().fit_transform(df_minmax[df_minmax.columns])
        return df_minmax, df
    
    elif scaler == "logstandardize":
        df_log_std = df.copy()
        for c in df.columns:
            df_log_std[c] = np.log(df_log_std[c].replace(-np.inf, np.nan))
        df_log_std[df_log_std.columns] = StandardScaler().fit_transform(df_log_std[df_log_std.columns])
        return df_log_std, df
    
    elif scaler == "none":
        df_scaled = df.copy()
        return df_scaled, df

def det_term_picker(dominant_term):                         # Simple function that returns the dominant deterministic term in order. Constant will always be second most dominant!
    if dominant_term == "Constant":
        coint_terms = {"Constant":0, "Trend":1, "None":2}
        johansen_terms = [0,1,-1]
        eg_terms = ["c", "ct", "nc"]
        vecm_terms = ["ci", "cili", "nc"]
    elif dominant_term == "Trend":
        coint_terms = {"Trend":0, "Constant":1, "None":2}
        johansen_terms = [1,0,-1]
        eg_terms = ["ct","c", "nc"]
        vecm_terms = ["cili","ci", "nc"]
    elif dominant_term == "None":
        coint_terms = {"None":0, "Constant":1, "Trend":2}
        johansen_terms = [-1,0,1]
        eg_terms = ["nc","c", "ct"]
        vecm_terms = ["nc","ci", "cili"]
    return coint_terms, johansen_terms, eg_terms, vecm_terms

def model_creator(df_scaled, first_column, second_column, p_max, vecm_term, joh_term, eg_term, threshold_joh):                              # Function to create bivariate VECM models based on det. terms
    df_pair = pd.concat([df_scaled[first_column], df_scaled[second_column]], axis=1)                                                        # Create a df consistent of only one pair of coins
    lag_order = vecm.select_order(df_pair, maxlags=p_max, deterministic=vecm_term)                                                          # Determine the optimal lag length based on AIC
    trace_rank = select_coint_rank(df_pair, det_order = joh_term, k_ar_diff = lag_order.aic, method = 'trace', signif=threshold_joh)        # Calculate the trace method statistic of johansen test
    score, p_value_eg, _ = coint(df_scaled[first_column], df_scaled[second_column], trend=eg_term, maxlag=lag_order.aic, autolag=None)      # Calculate the p value of engle-granger method
    return df_pair, lag_order, trace_rank, p_value_eg                                                                                       # Return the calculate cointegration scores 

def alpha_checker(df_pair, df_pairs, first_column, second_column, index, lag_order, vecm_terms, term_number, threshold_alpha, first):       # Function to check the alphas (error correction term) estimated by the VECM
    threshold_met = False                                                                                                               
    fitted_model = vecm.VECM(df_pair, k_ar_diff=lag_order.aic, coint_rank=1, deterministic=vecm_terms[term_number]).fit()                   # Estimate the VECM based on the det. term given
    if fitted_model.pvalues_alpha[0] < threshold_alpha and fitted_model.pvalues_alpha[1] < threshold_alpha:                                 # If both estimated alphas are significant, proceed
        if (fitted_model.alpha[0][0] < 0 and fitted_model.alpha[1][0] > 0) or (fitted_model.alpha[0][0] > 0 and fitted_model.alpha[1][0] < 0):  # If one alpha is negative and one is positive (mean reversion), proceed
            if first == True:
                if (fitted_model.alpha[0][0] < 0 and fitted_model.alpha[1][0] > 0):
                    alpha_one_neg = True
                elif (fitted_model.alpha[0][0] > 0 and fitted_model.alpha[1][0] < 0):
                    alpha_one_neg = False
                column_name = first_column + "-" + second_column                                                                            # If both statements hold true, return the beta coefficient estimated by the VECM
                betas = {"beta_0":fitted_model.const_coint[0][0], "beta_1":fitted_model.beta[1][0], "beta_2":fitted_model.lin_trend_coint[0][0], "alpha_one_neg":alpha_one_neg}
                df_pairs.loc[index, column_name] = [betas]                                                                                  # Beta coefficients account for the cointegration relationship.
                threshold_met = True                                                                                                        # If there is no trend/constant, this coefficient will return zero
            
            elif first == False:
                if (fitted_model.alpha[0][0] < 0 and fitted_model.alpha[1][0] > 0):
                    alpha_one_neg = True
                elif (fitted_model.alpha[0][0] > 0 and fitted_model.alpha[1][0] < 0):
                    alpha_one_neg = False
                column_name = second_column + "-" + first_column                                                                            # Also check the other way around if this is given. Columns will be switched
                betas = {"beta_0":fitted_model.const_coint[0][0], "beta_1":fitted_model.beta[1][0], "beta_2":fitted_model.lin_trend_coint[0][0], "alpha_one_neg":alpha_one_neg}
                df_pairs.loc[index, column_name] = [betas]
                threshold_met = True                                                                                                        # If threshold is met, it returns True and the for loop outside the function will break
        else:
            column_name = first_column + "-" + second_column                                                                                # If the threshold is not met, it returns False and the for loop outside continues
            df_pairs.loc[index, column_name] = np.nan                                                                                       # The for loop outside the function continues by plugging in different det. terms
            threshold_met = False
    else:
        column_name = first_column + "-" + second_column
        df_pairs.loc[index, column_name] = np.nan
        threshold_met = False
    
    return threshold_met, df_pairs                        # Return the dataframe containing the cointegrated pairs and the pairs that have NaN --> no cointegration

def adf_coint_tester(df_scaled, df_pairs, index, first_column, adf_term, threshold_adf, threshold_eg, threshold_joh, threshold_alpha, coint_terms, johansen_terms, vecm_terms, eg_terms, strategy_num):
    dftest = adfuller(df_scaled[first_column], regression = adf_term)                                       # Use the det. term given for the ADF test - Could be important parameter.
    p_max = int(12*(int(len(df_scaled))/100)**0.25)                                                         # Calculate the maximum number of lags to include in the models, according to Ng and Perron (1995)
    if dftest[1] < threshold_adf:                                                                           # Using ADF test: if time series in 10000 sec. is integrated at order 0, no cointegration is possible
        for second_column in df_scaled.columns:
            column_check = second_column + "-" + first_column
            if first_column != second_column and column_check not in list(df_pairs.columns):                # Skip the columns with double names (XRP-XRP) and columns that are already in (XRP-BTC) == (BTC-XRP).
                column_name = first_column + "-" + second_column
                df_pairs.loc[index, column_name] = np.nan                                                   # Return NaN meaning no cointegration possible between the two time series.
                
    else:
        for second_column in df_scaled.columns:                                                             # If the time series are integrated at order 1, cointegration is possible
            column_check = second_column + "-" + first_column                                               
            if first_column != second_column and column_check not in list(df_pairs.columns):                # Again skip the double names and double columns
                #set different types of deterministic terms as dominant                                     # Loop over the three different deterministic terms. If one is found significant cointegrated, break the loop for that det. term
                for term, term_number in coint_terms.items():                                               # Important to put dominant det. term first. Grid Search can be used in different combinations.
                    df_pair, lag_order, trace_rank, p_value_eg = model_creator(df_scaled, first_column, second_column, 
                                                                                p_max, vecm_terms[term_number], johansen_terms[term_number],        # Create the VECM and calculate the cointegration scores
                                                                                eg_terms[term_number], threshold_joh)
                    
                    if (trace_rank.test_stats[0] > trace_rank.crit_vals[0]):                                                                        # If johansen coint statistic exceeds the crit value, (XRP->BTC) proceed
                        if p_value_eg < threshold_eg:                                                                                               # If EG coint statistic is lower than the p_value threshold, proceed
                            df_pair_opp, lag_order, trace_rank_opp, p_value_eg_opp = model_creator(df_scaled, second_column, first_column,          # Create the VECM and test cointegration in the opposite direction
                                                                                                    p_max, vecm_terms[term_number], johansen_terms[term_number], 
                                                                                                    eg_terms[term_number], threshold_joh)
                        
                            if (trace_rank_opp.test_stats[0] > trace_rank_opp.crit_vals[0]):                                                        # If johansen coint statistic the opposite way is significant (BTC->XRP)
                                if p_value_eg_opp < threshold_eg:                                                                                   # Same goes for EG, opposite way
                                    if strategy_num == 3:                                                                                           # If this point is reached, significant cointegration is established
                                        threshold_met, df_pairs = alpha_checker(df_pair, df_pairs, first_column,                                    # For strategy 3; check the alpha coefficient on sign and significance
                                                                                second_column, index, lag_order, vecm_terms, 
                                                                                term_number, threshold_alpha, True)
                                        if threshold_met == True:                                                                                   # Break for loop if alpha coefficients are right sign and significant
                                            break
                                        
                                        else:                                                                                                       # If the alpha's were not significant, check the alphas opposite way
                                            threshold_met, df_pairs = alpha_checker(df_pair_opp, df_pairs, first_column,                            # This could deliver more than 15 columns.
                                                                                    second_column, index, lag_order, vecm_terms,                    # In this case XRP-BTC not equal to BTC - XRP
                                                                                    term_number, threshold_alpha, False)                            # Then trading can be done in the opposite relation
                                            if threshold_met == True:
                                                break
                                    
                                    elif strategy_num == 2:                                                                                         # For strategy 2 only significant cointegration needs to be established
                                        fitted_model = vecm.VECM(df_pair, k_ar_diff=lag_order.aic, coint_rank=1, deterministic=vecm_terms[term_number]).fit()
                                        column_name = first_column + "-" + second_column
                                        betas = {"beta_0":fitted_model.const_coint[0][0], "beta_1":fitted_model.beta[1][0], "beta_2":fitted_model.lin_trend_coint[0][0], "alpha_one_neg":True}
                                        df_pairs.loc[index, column_name] = [betas]                                                                  # Return the beta coefficients if there is significant cointegration
                                        break
                                    
                                    elif strategy_num == 1:
                                        fitted_model = sm.OLS(df_pair[first_column], df_pair[second_column]).fit()
                                        beta = fitted_model.params.values[0]
                                        column_name = first_column + "-" + second_column
                                        df_pairs.loc[index, column_name] = beta
                                        break                                                                     

                                else:
                                    column_name = first_column + "-" + second_column
                                    df_pairs.loc[index, column_name] = np.nan
                            else:
                                column_name = first_column + "-" + second_column
                                df_pairs.loc[index, column_name] = np.nan
                        else:
                            column_name = first_column + "-" + second_column
                            df_pairs.loc[index, column_name] = np.nan
                    else:
                        column_name = first_column + "-" + second_column
                        df_pairs.loc[index, column_name] = np.nan
    return df_pairs

def trade(S1, S2, resid, std_dev, position_one_usd, position_two_usd, threshold_1, threshold_2, stop_loss_one, second_stop, transaction_cost):
    position = position_one_usd                                             # USD - Capital to start your first and then second long-short position
    position_2 = position_two_usd                                           
    first_thresh = (threshold_1*std_dev)
    second_thresh = (threshold_2*std_dev)
    stop_loss = (stop_loss_one*std_dev)
    cost = transaction_cost
    money = 0                                                               # The amount of money you need to start the position. This is 0 since there are no transaction costs accounted for.
    trade_made = []
    countS1 = 0                                                             # The quantity of coin 1 and coin 2 in your wallet
    countS2 = 0                                                             
    position_taken = False                                                  # If position is taken, this will be true. 
    long_short_taken = False                                                # Same goes for the more specific positions. If long-short position is taken this binary string and position_taken = True 
    short_long_taken = False
    second_long_short_taken = False                                         # If a second position is taken when threshold 2 is exceded, this will return True
    second_short_long_taken = False

    for second in range(second_stop, len(resid)):                         # Iterate over the row of the residuals to check whether a residual exceeds the threshold.
        if short_long_taken == True:                                        # If a short_long postion is taken, it checks whether it needs to stop this position. 
            if resid[second] <= stop_loss:                                  # Exit position if current residual is lower than 0
                short_long_taken = False                                    
                position_taken = False                                      
                money += ((S1[second] * countS1) + (S2[second] * countS2) - cost)       # Determine the money made/lost when you sell/re-buy the quantity of crypto coins in your wallet at the exit
                if ((S1[second] * countS1) + (S2[second] * countS2) - cost) > 0:
                    win = 1
                    trade_made.append(win)
                elif ((S1[second] * countS1) + (S2[second] * countS2) - cost) < 0:
                    loss = 0
                    trade_made.append(loss)
                countS1 = 0                                                 # Return quantity in wallet zero after exiting your position
                countS2 = 0
            
            elif second_short_long_taken == False:                          # If first position is taken but second position is not taken, get in when residuals are exceding threshold 2, indicating extreme deviations. 
                if resid[second] > second_thresh:                           
                    second_short_long_taken = True                          # Take in the second position
                    countS1 -= position_2/S1[second]                        # Buy en Sell crypto with USD given for position 2. That is, you buy and sell the same amount of USD worth of crypto coins to start position. 
                    countS2 += position_2/S2[second]                        # Short - Long, sell S1 and buy S2. Creating an even position.

        elif long_short_taken == True:                                      # This statement considers the same arguments as above, but now for the long - short position. Buying S1 and selling S2 when residuals are negative.
            if resid[second] >= stop_loss:                                 
                long_short_taken = False
                position_taken = False
                money += ((S1[second] * countS1) + (S2[second] * countS2) - cost)
                if ((S1[second] * countS1) + (S2[second] * countS2) - cost) > 0:
                    win = 1
                    trade_made.append(win)
                elif ((S1[second] * countS1) + (S2[second] * countS2) - cost) < 0:
                    loss = 0
                    trade_made.append(loss)
                countS1 = 0
                countS2 = 0
            
            elif second_long_short_taken == False:                          
                if resid[second] < -second_thresh:
                    second_long_short_taken = True
                    countS1 += position_2/S1[second]
                    countS2 -= position_2/S2[second]
        
        elif position_taken == False:                                       # If no position is open, the first position can be opened if residuals exceed absolute value of 1.5 
            if resid[second] > first_thresh:                                # If a position is taken, it is not possible to take in another position unless you exit.
                short_long_taken = True                                     
                countS1 -= position/S1[second]
                countS2 += position/S2[second]
                
            elif resid[second] < -first_thresh:
                long_short_taken = True
                countS1 += position/S1[second]
                countS2 -= position/S2[second]

    return money, short_long_taken, long_short_taken, countS1, countS2, trade_made    #Return money made in this period, the position taken in case the position is still open, the number of coins in the wallet in case a position is open.

def trade_close(S1, S2, residuals, countS1, countS2, short_long_taken, long_short_taken, stop_loss_one, transaction_cost):
    money = 0                                                                   # Function to exit a position in the next period when the position was still open in the previous one.
    stop_loss = stop_loss_one
    cost = transaction_cost
    second_stop = 0 
    trade_made = []                                             
    for second in range(len(residuals)):                                     
        if short_long_taken == True:                                
            if residuals[second] <= stop_loss:                               
                short_long_taken = False                                        # Return short - long position on false to exit the position taken in the previous period (10000 sec.)
                money += ((S1[second] * countS1) + (S2[second] * countS2) - cost)              # Calculate money gained in this position, by selling the crypto that were still in the wallet. 
                if ((S1[second] * countS1) + (S2[second] * countS2) - cost) > 0:
                    win = 1
                    trade_made.append(win)
                elif ((S1[second] * countS1) + (S2[second] * countS2) - cost) < 0:
                    loss = 0
                    trade_made.append(loss)
                countS1 = 0                                                     
                countS2 = 0
                second_stop = second                                            # Remember the second in which the position is exited. The function will return this second and plug it in the next loop over residuals. 
                                                                                # This makes sure that no new position can be taken in the time that the position that exceeded the period was open.
        elif long_short_taken == True:                                          
            if residuals[second] >= stop_loss:                                  # Same goes for long - short position.
                long_short_taken = False
                money += ((S1[second] * countS1) + (S2[second] * countS2) - cost)  
                if ((S1[second] * countS1) + (S2[second] * countS2) - cost) > 0:
                    win = 1
                    trade_made.append(win)
                elif ((S1[second] * countS1) + (S2[second] * countS2) - cost) < 0:
                    loss = 0
                    trade_made.append(loss)
                countS1 = 0
                countS2 = 0
                second_stop = second
    
    return money, second_stop, trade_made                                                  # Return money made in closing the position that was still open and return the second at which the position was closed.                                                 # Return money made in closing the position that was still open and return the second at which the position was closed.

def pair_seperator(df, df_std, column_name):                                    # Function to seperate the coins in the column name and return the original time series.
    pair = column_name.split("-")   
    S1 = df[pair[0]]
    S2 = df[pair[1]]
    S1_std = df_std[pair[0]]                                                    # Return the coin of a standardized time series, standardized over #seconds interval (10000 used in research paper).
    S2_std = df_std[pair[1]]    
    time = np.array(range(0, len(S1_std)))                                      # Calculate the time to include for the linear trend in the cointegration relationship.
    
    return S1, S2, S1_std, S2_std, time

def back_tester(df_betas, df_input, seconds, position_one_usd, position_two_usd, threshold_1, threshold_2, stop_loss_one, scaler, strategy, transaction_cost):
    intervals = int(len(df_input)/seconds)                                                          # Determine the number of intervals, based on the seconds given in the function
    df_strategy = pd.DataFrame(index = range(1,(intervals+1)), columns = df_betas.columns)          # Create a new dataframe in which all the returns of that period are stored. Returns are made over one period later (index + 1)
    n_obs = int(len(df_input)/intervals)                                                            # Store the number of observations per interval. This is approx 10000 seconds. (10164 seconds)
    wallet = []                                                                                     # Create list in which the money is stored
    trades_made = {}
    for column_name in df_betas.columns:                                                            
        gains = 0                                                                                   # Set initial gains per column equal to zero
        second_stop = 0                                                                             # Set initial second_stop per column equal to zero, meaning there were no positions open before this point.
        for index,row in df_betas.iterrows():                                                       # Iterate over the number of intervals
            gains_period = 0                                                                        # Create a variable to store the gains per period. Which will reset every period
            if pd.isna(df_betas.loc[index, column_name]) == False:                                  # If the cell contains values for beta_0, beta_1 or beta_2, use these values to trade
                if strategy == 1:
                    df_scaled, df = dataframe_scaler(df_input, (index+1), n_obs, scaler)
                    df_scaled_past, df_past = dataframe_scaler(df_input, index, n_obs, scaler)
                    
                    S1, S2, S1_std, S2_std, time_cur = pair_seperator(df, df_scaled, column_name)
                    S1_past, S2_past, S1_past_std, S2_past_std, time_past = pair_seperator(df_past, df_scaled_past, column_name)
                    
                    scaled_spread = S1_std - df_betas.loc[index, column_name] * S2_std
                    scaled_spread_past = S1_past_std - df_betas.loc[index, column_name]*S2_past_std
                    std_dev_spread = np.std(scaled_spread_past)
                    pair = column_name.split("-")   
                    S1 = df[pair[0]]
                    S2 = df[pair[1]]
                    time = np.array(range(0, len(S1)))  
                    
                    profit, short_long_taken, long_short_taken, countS1, countS2 = trade(S1, S2, scaled_spread, std_dev_spread, position_one_usd, position_two_usd, threshold_1, threshold_2, stop_loss_one, second_stop, transaction_cost)
                    #profit, short_long_taken, long_short_taken, countS1, countS2 = trade2(df_input, df_betas, column_name, scaled_spread, index, n_obs, position_one_usd, position_two_usd, threshold_1, threshold_2, stop_loss_one, second_stop, transaction_cost)
                    gains += profit 
                    gains_period += profit

                    if short_long_taken == True or long_short_taken ==True:
                        if index != (intervals - 1):                                     
                            df_scaled_future, df_future = dataframe_scaler(df_input, (index + 2), n_obs, scaler)    
                            S1_future, S2_future, S1_future_std, S2_future_std, time_future = pair_seperator(df_future, df_scaled_future, column_name)

                            scaled_spread_future = S1_future_std - S2_future_std
                            profit, second_stop, trade_gains = trade_close(S1_future,S2_future,scaled_spread_future, countS1, countS2, short_long_taken, long_short_taken, stop_loss_one, transaction_cost)
                            column_trade = str(column_name) + "-" + str(index)
                            trades_made[column_trade] = trade_gains
                            gains += profit                                                                         
                            gains_period += profit

                elif strategy == 2 or strategy == 3:    
                    df_scaled, df = dataframe_scaler(df_input, (index+1), n_obs, scaler)                        # Trading will be done over the next period data
                    df_scaled_past, df_past = dataframe_scaler(df_input, index, n_obs, scaler)                  # Construct the df of the past period in order to determine the standard deviation of the residuals in the past period.

                    if df_betas.loc[index, column_name]['alpha_one_neg'] == True:
                        S1, S2, S1_std, S2_std, time_cur = pair_seperator(df, df_scaled, column_name)               # Seperate original, standardized time series of the seperate coins
                        S1_past, S2_past, S1_past_std, S2_past_std, time_past = pair_seperator(df_past, df_scaled_past, column_name)
                    elif df_betas.loc[index, column_name]['alpha_one_neg'] == False:
                        S2, S1, S2_std, S1_std, time_cur = pair_seperator(df, df_scaled, column_name)               # Seperate original, standardized time series of the seperate coins
                        S2_past, S1_past, S2_past_std, S1_past_std, time_past = pair_seperator(df_past, df_scaled_past, column_name)
                    
                    # Calculate the residuals, current and past. Residuals of current period are determined using the beta coefficients estimated in the VECM of past period data
                    resid_current = S1_std - df_betas.loc[index, column_name]['beta_0'] - (df_betas.loc[index, column_name]['beta_1']*S2_std) - (df_betas.loc[index, column_name]['beta_2']*time_cur)  
                    resid_past = S1_past_std - df_betas.loc[index, column_name]['beta_0'] - (df_betas.loc[index, column_name]['beta_1']*S2_past_std) - (df_betas.loc[index, column_name]['beta_2']*time_past)
                    std_dev_resid = np.std(resid_past)
                    
                    # Trade using the residuals as measure of deviations between the two time series S1 and S2
                    profit, short_long_taken, long_short_taken, countS1, countS2, trade_gains = trade(S1, S2, resid_current, std_dev_resid, position_one_usd, position_two_usd, threshold_1, threshold_2, stop_loss_one, second_stop, transaction_cost)
                    gains += profit                                                                             # Add the profit or loss to the gains made per column
                    gains_period += profit                                                                      # Add the profit or loss to the gains made per column and per period
                    column_trade = str(column_name) + "-" + str(index)
                    trades_made[column_trade] = trade_gains

                    if short_long_taken == True or long_short_taken ==True:                                     # If a position was still open during the last trading period. Use the same coefficients to trade in period + 1
                        if index != (intervals - 1): 
                            df_scaled_future, df_future = dataframe_scaler(df_input, (index + 2), n_obs, scaler)    # To trade in the future, create df's that contain data of the coins in period + 1 (future is index +2)
                            S1_future, S2_future, S1_future_std, S2_future_std, time_future = pair_seperator(df_future, df_scaled_future, column_name)

                            resid_future = S1_future_std - df_betas.loc[index, column_name]['beta_0'] - (df_betas.loc[index, column_name]['beta_1']*S2_future_std) - (df_betas.loc[index, column_name]['beta_2']*time_future)
                            profit, second_stop, trade_gains = trade_close(S1_future,S2_future,resid_future, countS1, countS2, short_long_taken, long_short_taken, stop_loss_one, transaction_cost)
                            gains += profit                                                                         # Add the profit or loss, that is made on the position that was still open, to the gains made per column
                            gains_period += profit                                                                  # Add the profit or loss, that is made on the position that was still open, to the gains made per column and per period
                            column_trade = str(column_name) + "-" + str(index) + "-" + "closed"
                            trades_made[column_trade] = trade_gains 
                                                                                                                # Return the second at which the position was closed. So in the next period, it cannot trade before this second.
            df_strategy.loc[(index+1),column_name] = gains_period                                               # Add the gains per columns/ per period to the dataframe.

        
        wallet.append(gains)                                                                                    # Append all the gains per column to the list. That will give the total gains per column
    return np.sum(wallet), df_strategy, trades_made                                                                          # Sum the gains of all the columns to give the total gains of that month

def strategy_tester(df_input, seconds, begin, strategy, scaler, dominant_term, d_term_adf, threshold_adf, threshold_eg, threshold_joh, threshold_alpha):
    coint_terms, johansen_terms, eg_terms, vecm_terms = det_term_picker(dominant_term)                      # Function that will test the strategy as a whole. First produce the dominant deterministic term

    intervals = int(len(df_input)/seconds)                                                                  
    df_pairs = pd.DataFrame(index = range(begin,intervals))
    n_obs = int(len(df_input)/intervals)

    for index,row in df_pairs.iterrows():                                                                   # Iterate over every row in the dataframe that has all intervals for rows (0-17) for research
        df_scaled, df = dataframe_scaler(df_input, index, n_obs, scaler)                                    # Scale every data frame of approx 10000 sec. that is created 

        for columns in df_scaled.columns:                                                                   # Iterate over every coin in the dataframe to produce pairs
            df_pairs = adf_coint_tester(df_scaled, df_pairs, index, columns, d_term_adf, threshold_adf, threshold_eg, threshold_joh, threshold_alpha, coint_terms, johansen_terms, vecm_terms, eg_terms, strategy)
    
    return df_pairs                                                                                                     # Use the pairs dataframe to trade on the pairs that are cointegrated
    

coins = {'XRP':4500, 'LTC':10, 'ADA':1500, 'SOL':150, 'BTC':0.04, 'ETH':1}  #Coins are placed in order including a quantity threshold of minimum 2000 USD worth of quantity based on March average price.
seasons = {"march":"03", "june":"06", "sept":"09", "dec":"12"}             #put the names of the months in the list, based on their name that is stored in the h5 file.

days = 2                                                                    #All data analysis for first two research question are done on 2 days of data.
frequency = "S"        

df_prices = {}                                                              #Store the deep prices of every month and every coin in a dictionary
df_prices_selected_days = {}
for month, day in seasons.items():                                          #Iterate over the months in the list seasons
    #df_prices[month] = deep_price(df_creator(month), 0, 10, 20, coins)     #Calculate the deep price using H5 files, giving the first column of price, the first column of quantity and the split point between bid and ask.
    df_prices[month] = pd.read_csv("lib/df_" + month + ".csv", index_col=0)     #Read in the deep prices from csv files calculated earlier.
    df_prices_selected_days[month] = sample_processor(df_prices[month], days, frequency, "2021-"+day+"-20")

seconds_list = [5000,10000,15000]                                           # Seconds intervals
threshold_list = {1:[0.01,0.1,0.0],2:[0.5,1,0.1], 3:[1.5,3,0.5],4:[2,4,1]}  # firsth threshold, second threshold, stoploss
transaction_cost_list = [0,1,2]                                             # In USD
pvalue_list = [0.1,0.05,0.01]                                               # P value for all statistical tests in the process
strategy_list = [2,3]                                                       # 2:unrestricted cointegration based strategy and 3: restricted cointegration based strategy
deterministic_list = ["Constant", "Trend", "None"]


df_pairs = strategy_tester(df_prices_selected_days[list(seasons.keys())[3]], seconds=seconds_list[0], 
                                        begin=0, strategy=strategy_list[1], scaler="standardize", # "none" for no scaling
                                        dominant_term=deterministic_list[2], d_term_adf="c",      
                                        threshold_adf=pvalue_list[0], threshold_eg=pvalue_list[0], 
                                        threshold_joh=pvalue_list[0], threshold_alpha=pvalue_list[0])

gains, df_gains_strategy, df_win_loss = back_tester(df_pairs, df_prices_selected_days[list(seasons.keys())[3]], seconds=seconds_list[0], 
                                        position_one_usd=2000, position_two_usd=4000, 
                                        threshold_1= list(threshold_list.items())[0][1][0], threshold_2= list(threshold_list.items())[0][1][1], 
                                        stop_loss_one= list(threshold_list.items())[0][1][2], scaler= "standardize", 
                                        strategy= strategy_list[1], transaction_cost= transaction_cost_list[0])

