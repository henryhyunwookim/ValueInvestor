# ValueInvestor

### <b>Background</b>

We are a portfolio investment company and we make investments in the emerging markets around the world. Our company profits by investing in profitable companies, buying, holding and selling company stocks based on value investing principles.

Our goal is to establish a robust intelligent system to aid our value investing efforts using stock market data. We make investment decisions and based on intrinsic value of companies and do not trade on the basis of daily market volatility. Our profit realization strategy typically involves weekly, monthly and quarterly performance of stocks we buy or hold.

### <b>Data Description</b>

You are given a set of portfolio companies trading data from emerging markets including 2020 Q1-Q2-Q3-Q4 2021 Q1 stock prices. Each company stock is provided in different sheets. Each market's operating days varies based on the country of the company and the market the stocks are exchanged. Use only 2020 data and predict with 2021 Q1 data.

### <b>Goal</b>

Predict stock price valuations on a daily, weekly and monthly basis. Recommend BUY, HOLD, SELL decisions. Maximize capital returns, minimize losses. Ideally a loss should never happen. Minimize HOLD period.

### <b> Success Metric</b>

Evaluate on the basis of capital returns. Use Bollinger Bands to measure your systems effectiveness.

### <b> Results</b>

<u>Model Performance</u>

To forecast stock price valuations, we utilized three regression models - SARIMAX, XGB Regressor, and Random Forest Regressor. Through thorough training and evaluation, the SARIMAX model was chosen for our system for its superior performance. Our SARIMAX model demonstrated outstanding accuracy, boasting an average Mean Absolute Percentage Error (MAPE) of merely 0.017788 across 8 distinct stocks in various countries. This signifies a remarkable level of precision, as the model's predictions for diverse stocks were, on average, only 1-2% off from the actual stock prices.

<u>Capital Returns</u>

The performance of our system was evaluated by comparing capital returns with a typical Bollinger Bands trading strategy. The system achieved a gain of 494.12 in daily trading, a loss of 25.37 in weekly trading, and a gain of 1,130.0 in monthly trading. A slight loss was incurred in weekly trading due to the inherent limitations of our predictive model, which is designed to adapt to various stocks and trading markets rather than conform to a one-size-fits-all approach - a rarity in the dynamic landscape of stock trading and valuation. 

A fundamental challenge in comprehensively measuring capital returns stemmed from the disparate scales of stock prices across different companies, countries, and currencies. Despite this, our system exhibited superior performance when considered in its entirety, surpassing the effectiveness of a conventional Bollinger Bands strategy. This achievement is particularly significant, given the widespread utilization of Bollinger Bands as an effective trading strategy. It's worth noting that the modeling and evaluation procedures were conducted with limited information on each company and the specific trading periods, further highlighting the complexity and inherent challenges of predicting stock prices and making informed trading decisions.

<u>Future Work</u>

In future iterations of this project, our primary objective would be to enrich the dataset with a broader range of data sources, including but not limited to financial data and news articles pertaining to specific companies. These additional data points will be seamlessly integrated into the system as needed, enhancing its analytical capabilities. Furthermore, it's essential to recognize that the magnitude of capital returns can fluctuate significantly based on various factors, including the initial balance and number of stocks, different window sizes and standard deviations for the Bollinger Bands, and the number of stocks to trade on each trading date. Therefore, an in-depth exploration of these variables is imperative to fortify and stabilize the trading system, ensuring its robustness and reliability in aiding trading decisions.

### <b>Notebook and Installation</b>

To run ValueInvestor.ipynb locally, please clone or fork this repo and install the required packages by running the following command:

pip install -r requirements.txt

##### Source: Apziva