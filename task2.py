"""You need to create a prototype pricing model that can go through further validation and testing 
before being put into production. Eventually, this model may be the basis for fully automated quoting 
to clients, but for now, the desk will use it with manual oversight to explore options with the client. 

You should write a function that is able to use the data you created previously to price the contract.
The client may want to choose multiple dates to inject and withdraw a set amount of gas, so your 
approach should generalize the explanation from before. Consider all the cash flows involved in the product.

The input parameters that should be taken into account for pricing are:

Injection dates. 
Withdrawal dates.
The prices at which the commodity can be purchased/sold on those dates.
The rate at which the gas can be injected/withdrawn.
The maximum volume that can be stored.
Storage costs.

Write a function that takes these inputs and gives back the value of the contract. 
You can assume there is no transport delay and that interest rates are zero. Market holidays, 
weekends, and bank holidays need not be accounted for. Test your code by selecting a few sample inputs."""

def pricing_the_contract(injection, withdrawal, prices, rates, max_volume, storage_cost):
    return value_of_contract