import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('FuelConsumption.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data Visualization
plt.figure(figsize=(15, 10))

# 1. Distribution of CO2 Emissions
plt.subplot(2, 2, 1)
sns.histplot(df['CO2EMISSIONS'], bins=30, kde=True)
plt.title('Distribution of CO2 Emissions')
plt.xlabel('CO2 Emissions (g/km)')

# 2. Fuel Consumption by Vehicle Class
plt.subplot(2, 2, 2)
sns.boxplot(x='VEHICLECLASS', y='FUELCONSUMPTION_COMB', data=df)
plt.title('Fuel Consumption by Vehicle Class')
plt.xticks(rotation=45)
plt.ylabel('Combined Fuel Consumption (L/100km)')

# 3. Engine Size vs CO2 Emissions
plt.subplot(2, 2, 3)
sns.scatterplot(x='ENGINESIZE', y='CO2EMISSIONS', data=df, hue='CYLINDERS')
plt.title('Engine Size vs CO2 Emissions')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')

# 4. Top 10 Makes by Average Fuel Consumption
plt.subplot(2, 2, 4)
top_makes = df.groupby('MAKE')['FUELCONSUMPTION_COMB'].mean().sort_values().head(10)
sns.barplot(x=top_makes.values, y=top_makes.index)
plt.title('Top 10 Most Fuel Efficient Makes')
plt.xlabel('Average Combined Fuel Consumption (L/100km)')

plt.tight_layout()
plt.show()

# Correlation analysis
correlation = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']].corr()
print("\nCorrelation Matrix:")
print(correlation)

# Fuel type analysis
fuel_type_analysis = df.groupby('FUELTYPE').agg({
    'FUELCONSUMPTION_COMB': 'mean',
    'CO2EMISSIONS': 'mean',
    'MAKE': 'count'
}).rename(columns={'MAKE': 'Count'})
print("\nFuel Type Analysis:")
print(fuel_type_analysis)

# Save cleaned data
df.to_csv('Cleaned_FuelConsumption.csv', index=False)
print("\nCleaned data saved to 'Cleaned_FuelConsumption.csv'")