Note: This project is about apple stocks.

In this pytorch project, I use one data set, but i feed in two different models different x and y.

Model 1 has the x of the previous prices and the y of what the price will be.
Mode 2 will be the previous prices AND the moving average(trend) and y will be the price right now.
You can run the code to see which one runs better.
I used the yahoo databases to get the data.
Spoiler: Its obvious that Model 2 does better, because there is no way that a model can just predict it from the date.
