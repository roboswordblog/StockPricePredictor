import yfinance as yf
# This is to get all the stocks starting from the start of 2020
data = yf.download("AAPL", start="2020-01-01")
data.to_csv("aapl.csv")
