import csv
from scipy import optimize
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates

def main():
    # File path
    path = 'gas.csv'

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

    def pricing_the_contract(injection, withdrawal, rates, max_volume, storage_cost, inj_cost, withdraw_cost, transportation_cost):
        # Convert date strings to datetime objects and then to numeric format
        injection_dates = [mdates.date2num(datetime.strptime(date, "%m/%d/%Y")) for date in injection]
        withdrawal_dates = [mdates.date2num(datetime.strptime(date, "%m/%d/%Y")) for date in withdrawal]

        # Calculate total injected volume
        total_injected_volume = sum(rates)
        total_volume = min(max_volume, total_injected_volume)

        # Ensure the rates match the injected volume
        rates = [rate * total_volume / total_injected_volume for rate in rates]

        # Calculate buying and selling prices using the fitted model
        buying_prices = [seasonal_model_with_trend(date, *params) for date in injection_dates]
        selling_prices = [seasonal_model_with_trend(date, *params) for date in withdrawal_dates]

        # Calculate the total buying cost
        total_buying_cost = sum(price * rate for price, rate in zip(buying_prices, rates))

        # Calculate the total selling revenue
        total_selling_revenue = sum(price * rate for price, rate in zip(selling_prices, rates))

        # Calculate storage cost: assuming storage cost per day and converting it to the period between injection and withdrawal
        days_storage = (max(withdrawal_dates) - min(injection_dates)).astype(int)
        total_storage_cost = storage_cost * (days_storage / 30)  # Assuming 30 days per month

        # Calculate the number of full units of 1,000,000 in the total volume
        units = total_volume / 1000000

        # Calculate the total injection cost
        total_inj_cost = inj_cost * units

        # Calculate the total withdrawal cost
        total_withdraw_cost = withdraw_cost * units

        # Calculate the total cost for both injection and withdrawal
        total_inj_withdraw_cost = total_inj_cost + total_withdraw_cost
        
        transportation_cost = transportation_cost * len(injection_dates)
        # Calculate the value of the contract
        value_of_contract = total_selling_revenue - total_buying_cost - total_storage_cost - total_inj_withdraw_cost - transportation_cost
        return value_of_contract

    # Example Inputs
    inject = ["07/01/2022", "07/02/2022"]
    withd = ["12/23/2024", "12/25/2024"]
    rates = [1000000, 1000000]  # Injection/withdrawal rate in MMBtu
    max_volume = 20000000  # Maximum volume that can be stored in MMBtu
    storage_cost = 10000  # Cost per month (assuming a constant rate per month)
    inj_cost = 10000  # Injection cost per 1 million MMBtu
    withdraw_cost = 10000  # Withdrawal cost per 1 million MMBtu
    transp_cost = 50000

    # Calculate and print the value of the contract
    value = pricing_the_contract(inject, withd, rates, max_volume, storage_cost, inj_cost, withdraw_cost, transp_cost)
    print(f"Value of the contract: ${value:.2f}")

if __name__ == '__main__':
    main()