
Online Portfolio Selection
===========

The purpose of this package is to put together different online portfolio selection algorithms and provide unified tools for their analysis. If you do not know what online portfolio is, you can look at a recent [survey by Bin Li and Steven C. H. Hoi](http://arxiv.org/abs/1212.2129).

In short, the purpose of online portfolio is to choose portfolio weights in every period(per day, per minute) to maximize its final wealth. Examples of such portfolios could be simple buy-and-hold portfolio.

Currently, there is an active research in the area of online portfolios and even though its results are mostly theoretic, algorithms for practical use starts to appear. 

In this framework, I mainly implemented one algorithm: Confidence Weighted Mean Reversion (CWMR) Strategy. Details of the trading algorithm can be seen in the PDF file "Confidence Weighted Mean Reversion for Online Portfolio Selection".


## Installation
This trading framework is based on Python 3.7.

The required Python packages can be seen in requirements.txt file. You can install the packages one by one if you don't have them.

## Instructions
The following code run instructions are based on PyCharm. 

After choose the mainrun.py file to run. You need to edit configurations of this file. Mainly add the following parameters to the parameter box:

--model cwmr --data data --fees 0.0003

Parameter "model" means the trading algorithms you can choose. In this framework, we only have one option: "cwmr". You can also write your own trading algorithm and add it to the framework if you like. 

Parameter "data" means the stock price data for each trading period that we use. Based on the given dataset in the project. We choose the trading period as every day and use the last price data of every day.

Parameter "fees" means the transaction fees during the trading process. We assume proportional transaction costs on risky assets purchases and sales. Normally, we set the transaction fees rate as 0.03% for purchases and sales.

## Output and Performance Metrics
The initial wealth is assumed to be 1. After successfully running the code. It will print the portfolio(Each element corresponds to the stock ticker is the order of the stock ticker we selected in the PDF file) of each trading day, and the cumulative wealth at the end of every day. The final result is the final cumulative wealth at the end of total trading periods. 

Some performance metrics, like the Sharpe ratio and the Max drawdown, will also be shown.