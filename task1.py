import csv
from scipy import optimize
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
from scipy.signal import argrelextrema

def main():
    # File path
    path = r'/workspaces/JPQuantForage/gas.csv'

    # Initialize lists
    dates = []
    prices = []

    # Read CSV file
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            dates.append(datetime.strptime(row[0], "%m/%d/%y"))  # Keep dates as datetime objects
            prices.append(float(row[1]))  # Convert prices to float

    # Convert datetime objects to numeric format for fitting
    numeric_dates = mdates.date2num(dates)

    # Define the sinusoidal function with a linear trend
    def seasonal_model_with_trend(x, a, b, c, d, e):
        return a * np.sin(b * x + c) + d * x + e

    # Fit the sinusoidal model with a linear trend to the data
    params, params_covariance = optimize.curve_fit(seasonal_model_with_trend, numeric_dates, prices, p0=[1, 2*np.pi/(365.25), 0, 0, np.mean(prices)])

    # Plot data points and the fitted model
    plt.scatter(dates, prices, label="Data points")
    plt.plot(dates, seasonal_model_with_trend(numeric_dates, *params), color='red', label="Seasonal model with trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    # Format x-axis to show months
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Find local maxima and minima
    fitted_prices = seasonal_model_with_trend(numeric_dates, *params)
    maxima_indices = argrelextrema(fitted_prices, np.greater)[0]
    minima_indices = argrelextrema(fitted_prices, np.less)[0]
    extrema_indices = np.sort(np.concatenate((maxima_indices, minima_indices)))

    # Label the local maxima and minima
    for idx in extrema_indices:
        date = dates[idx]
        price = fitted_prices[idx]
        plt.annotate(f'{date.strftime("%b")}', xy=(date, price), xytext=(date, price + 0.5), ha='center')

    # Show plot
    plt.show()

    # Example input date prediction
    input_date = "07/01/2024"
    input_datetime = datetime.strptime(input_date, "%m/%d/%Y")
    input_numeric_date = mdates.date2num(input_datetime)
    predicted_price = seasonal_model_with_trend(input_numeric_date, *params)
    print(predicted_price)

if __name__ == '__main__':
    main()