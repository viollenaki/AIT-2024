import pandas as pd

df = pd.read_csv('products-100.csv')

df_selected = df[['Name', 'Brand', 'Category', 'Price', 'Currency', 'Stock', 'Availability']]

available_products = df_selected[df_selected['Availability'].isin(['in_stock', 'limited_stock'])]

avg_price = available_products['Price'].mean()
print(f"Average price of available products: {avg_price:.2f}")

high_value_products = available_products[available_products['Price'] > 500]
print(f"Number of high-value products: {len(high_value_products)}")

print("Characteristics of high-value products:")
print(high_value_products.describe())
