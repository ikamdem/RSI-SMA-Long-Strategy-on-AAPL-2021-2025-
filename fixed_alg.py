
import time
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from KEY import api_key

url_string = "https://www.alphavantage.co/query?"

# function to retrieve the daily prices
def retrieve_data(symbol, interval='TIME_SERIES_DAILY'): 
    parameters = {
        "function" : interval,
        "symbol" : symbol,
        "apikey" : api_key,
        "outputsize" : "full"}
    
    response = requests.get(url_string, params=parameters)
    data = response.json()

    time.sleep(60)

    if 'Time Series (Daily)' not in data:  
        raise ValueError('Invalid API response. Make sure the symbol is correct and try again.')
    
    ts_data = data["Time Series (Daily)"]
    return ts_data 

# function to transform retrieved data
def transform_data(ts_data):
    df = pd.DataFrame.from_dict(ts_data, orient="index")
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] 

    # Convert index to date
    df.index = pd.to_datetime(df.index)

    df = df.astype(float)

    return df 


symbol = "AAPL"

# fetching data
try:
    raw_data = retrieve_data(symbol)
    stock_df = transform_data(raw_data)

    print(stock_df.head())

except ValueError as e: 
    print(f'Error encountered: {e}')

# sort by date in ascending order
stock_df.sort_index(inplace=True)
df = stock_df.loc["2020-07-10" : "2025-07-10"]
print(df)

# create a column from day 1 to day n    
df["day"] = np.arange(1, len(df) + 1)

# function to compute RSI
def rsi(prices, period):
    delta = prices.diff()
    up = delta.clip(lower = 0)
    down = delta.clip(upper = 0, lower = None)

    ema_up = up.ewm(alpha=1/period, min_periods = period).mean()
    ema_down = down.abs().ewm(alpha=1/period, min_periods = period).mean()

    rs = ema_up / ema_down 

    rsi = 100 - 100 / (1 + rs)

    return rsi

# function to compute SMA
def sma(prices, window):
    sma = prices.rolling(window=window).mean()
    return sma

df["RSI"] = rsi(df["Close"], 14).shift(1)

df["SMA 20"] = sma(df["Close"], 20).shift(1)
df["SMA 50"] = sma(df["Close"], 50).shift(1)

# removing and rearranging columns
df.drop(["Open", "High", "Low", "Volume"], axis = 1, inplace=True)
df = df[["day", "Close", "RSI", "SMA 20", "SMA 50"]]

df.head()

# to avoid NA coming from indicators and stock split from APPLE in 2020
df_backup = df.copy()
df = df[df.index >= "2021-01-01"]
df

# conditions to enter a trade 
RSI_bool = df["RSI"] > 70 
SMA20_bool = (df["SMA 20"] > df["SMA 50"])
Close_bool = (df["Close"] > df["SMA 20"])

confidence_score = (
    RSI_bool * 1.5 +
    SMA20_bool * 0.5 +
    Close_bool * 1
)

df["Confidence Score"] = confidence_score
df["BuySignal"] = df["Confidence Score"] == 3


df["FilteredSignal"] = False # to get the day the trade was entered into
df["ExitTrade"] = False # to get the day the trade is closed

InTrade = False
for i in range(len(df)):
    if not InTrade and df["BuySignal"].iloc[i] == True : # entry point
        df.loc[df.index[i], "FilteredSignal"] = True
        InTrade = True

    elif InTrade and  df["SMA 20"].iloc[i] < df["SMA 50"].iloc[i]: # exit point
        df.loc[df.index[i], "ExitTrade"] = True
        InTrade = False


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex = True)

# PRICES + SMAs
ax1.plot(df.index, df["Close"], label="Close", linewidth=1)
ax1.plot(df.index, df["SMA 20"], label="SMA 20", linewidth=1)
ax1.plot(df.index, df["SMA 50"], label="SMA 50", linewidth=1)
ax1.set_ylabel("Price")
ax1.set_title("AAPL — Price and SMA")
ax1.legend()
ax1.grid(True)

# RSI 
ax2.plot(df.index, df["RSI"], color='blue', label="RSI")
ax2.axhline(70, color='red', linestyle='--', linewidth=0.8)   
ax2.axhline(30, color='green', linestyle='--', linewidth=0.8) 
ax2.set_ylabel("RSI")
ax2.set_title("RSI")
ax2.legend()
ax2.grid(True)


for date, row in df.iterrows():
    if row["FilteredSignal"] == True: # green arrow
        ax1.annotate(
            text="", 
            xy=(date, row["Close"]),          
            xytext=(date, row["Close"] + 5),     
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5)
        )
    if row["ExitTrade"] == True: # red arrow
        ax1.annotate(
            text="",
            xy=(date, row["Close"]),
            xytext = (date, row["Close"] - 5), 
            arrowprops = dict(arrowstyle="->",color = "red", lw=1.5)
        )

plt.tight_layout()
plt.show

# performance 
in_trade = False
results = []

for date, row in df.iterrows(): 
    if row["FilteredSignal"] == True : 
        entry_date = date
        entry_price = row["Close"]
        in_trade = True

    elif row["ExitTrade"] == True and in_trade: 
        exit_date = date
        exit_price = row["Close"]
        results.append((entry_date, entry_price, exit_date, exit_price))
        in_trade = False

# creation of a new df containing details of the trade
results_df = pd.DataFrame(results, columns=["EntryDate", "EntryPrice", "ExitDate", "ExitPrice"])
results_df["Return"] = (results_df["ExitPrice"] - results_df["EntryPrice"]) / results_df["EntryPrice"]
results_df

df2 = results_df.copy()

print(f"# total of trades is {len(df2)}")
count_win = 0
count_lose = 0
for index, row in results_df.iterrows():
    if row["Return"] < 0 : 
        count_lose += 1 
    if row['Return'] > 0 : 
        count_win += 1 

print(f"# of winning trades is {count_win}")
print(f"# of losing trades is {count_lose}")

 
count_win = (df2["Return"] > 0).sum() # easier possibility but I wanted to use .iterrows()
count_lose = (df2["Return"] < 0).sum()

# mean return of the strategy
mean_ret = df2["Return"].mean()
print(f"Mean return is {mean_ret:.3f}")

# success rate of the strategy
success_rate = count_win / len(df2)
print(f"Success rate: {success_rate:.2%}")


Io = 1000
df2["Capital"] = 1000
df2["Evolution"] = 0 
entry_fee = 25 # fixed fee

# compute the value of the capital with a list
evo_cap = []
current_value = df2["Capital"].iloc[0]
for i in range(len(df2)):
    current_value = current_value * (1 + df2.loc[df2.index[i], "Return"]) - entry_fee
    evo_cap.append(current_value)

df2["Evolution"] = evo_cap # assign the list created as the column "Evolution" (of the capital)


plt.plot(df2["EntryDate"], df2["Evolution"], label = "Capital")
plt.legend
plt.show

for i in range(len(df2)):
    entry = df2.loc[i, "EntryDate"]
    exit = df2.loc[i, "ExitDate"]
    capital = df2.loc[i, "Evolution"]
    
    mask = (df.index >= entry) & (df.index <= exit)
    df.loc[mask, "Capital"] = capital

df["Capital"] = df["Capital"].ffill() # fill the blank in the column "Capital" with the last value 

plt.plot(df.index, df["Capital"])

df["DailyReturn"] = df["Capital"].pct_change()
df.dropna(inplace=True)

returns = df["DailyReturn"]
returns = returns[returns != 0] # remove 0% returns that occur when no position is open
returns


def SR(returns, rf):
    mu = returns.mean()
    std = returns.std()
    return ((mu - (rf / 252)) / std) * np.sqrt(252)


SR(returns, 2) # this sharpe ratio is not pertinent given that returns are based only on a few discrete trades, not on continuous daily performance
