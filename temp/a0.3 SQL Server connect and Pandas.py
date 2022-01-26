''' SQL to Pandas exercise
'''

import pyodbc
import pandas as pd

#set variables
driver = 'SQL Server'
server = 'DESKTOP-AL-01'
database = 'Northwind'

#using Trusted Connection, using your Windows account
conn = pyodbc.connect('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+';Trusted_Connection=yes;')
cursor = conn.cursor()


#query the database
query = "SELECT TOP 50 [OrderID], [CustomerID], [EmployeeID], [OrderDate], [ShippedDate], [ShipCity], [ShipRegion], [ShipCountry] FROM dbo.Orders;"
df_orders = pd.read_sql(query, conn)
print(df_orders.head())


query = "SELECT [CustomerID], [CompanyName], [ContactName], [ContactTitle], [City], [Region],  [Country], [Phone] FROM dbo.Customers;"
df_customers = pd.read_sql(query, conn)
print(df_customers.head())

#======= SQL to Pandas exercise =======
''' resources: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html
    tuturials1: https://www.dataquest.io/blog/pandas-cheat-sheet/
    tuturials2: https://levelup.gitconnected.com/sql-v-pandas-basic-syntax-comparison-cheat-sheet-498289372d45
    tuturials3: https://www.educative.io/blog/pandas-cheat-sheet
'''


#merge or similar is join
df_orders.merge(df_customers, left_on='CustomerID', right_on='CustomerID', how='inner', suffixes=('_left', '_right'))
df_orders_customers = df_orders.merge(df_customers, left_on='CustomerID', right_on='CustomerID', how='inner', suffixes=('_x', '_y'), sort=True)
df_orders_customers.describe()
df_orders_customers.info()



