# Blocktraders_cointegrated_pairstrading
This repository contains the script and data regarding the cointegrated based pairs trading strategy. The strategy is developed under supervision of Blocktraders for completion of the Research Project course at the Vrije Universiteit.

NOTE
Specific parameters are in place to run the strategy in a certain time frame under certain circumstances. 
- Time Interval: 5000
- Scaling Method: Standardisation
- P-value for statistics: 0.1
- Strategy: Restricted Cointegration Based
- First Threshold: 0.01 stdev
- Second Threshold: 0.1 stdev
- Stoploss: 0.0 stdev
- Transaction cost: 0$

This is easily adjustable using different indices in the function “strategy_tester” and “back_tester”.

