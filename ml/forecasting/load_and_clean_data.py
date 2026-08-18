from pathlib import Path
import pandas as pd

DATA_DIR = Path("ml/data/raw/benroshan")
OUTPUT_DIR = Path("ml/data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


orders = pd.read_csv(DATA_DIR / "List of Orders.csv")
details = pd.read_csv(DATA_DIR / "Order Details.csv")

print("=" * 80)
print("STEP 1 : DATA LOADED")
print("=" * 80)

print(f"Orders Shape : {orders.shape}")
print(f"Details Shape: {details.shape}")


orders = orders.dropna(
    subset=[
        "Order ID",
        "Order Date",
        "CustomerName",
        "State",
        "City"
    ]
)

details = details.dropna()

print("\nAfter Cleaning")

print(f"Orders Shape : {orders.shape}")
print(f"Details Shape: {details.shape}")



orders = orders.drop_duplicates()

details = details.drop_duplicates()

print("\nAfter Removing Duplicates")

print(f"Orders Shape : {orders.shape}")
print(f"Details Shape: {details.shape}")


orders["Order Date"] = pd.to_datetime(
    orders["Order Date"],
    dayfirst=True,
    errors="coerce"
)

details["Amount"] = pd.to_numeric(details["Amount"])
details["Profit"] = pd.to_numeric(details["Profit"])
details["Quantity"] = pd.to_numeric(details["Quantity"])

# Remove invalid dates

orders = orders.dropna(subset=["Order Date"])


orders = orders.sort_values("Order Date")



merged = pd.merge(
    orders,
    details,
    on="Order ID",
    how="inner"
)



print("\n" + "=" * 80)
print("VALIDATION REPORT")
print("=" * 80)

print("\nDate Range")

print(merged["Order Date"].min())
print(merged["Order Date"].max())

print("\nUnique Orders")

print(merged["Order ID"].nunique())

print("\nUnique Customers")

print(merged["CustomerName"].nunique())

print("\nUnique States")

print(merged["State"].nunique())

print("\nUnique Cities")

print(merged["City"].nunique())

print("\nCategories")

print(merged["Category"].value_counts())

print("\nSub Categories")

print(merged["Sub-Category"].nunique())

print("\nTotal Sales")

print(f"₹ {merged['Amount'].sum():,.2f}")

print("\nTotal Profit")

print(f"₹ {merged['Profit'].sum():,.2f}")

print("\nTotal Quantity")

print(int(merged["Quantity"].sum()))



merged.to_csv(
    OUTPUT_DIR / "forecasting_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

print(
    OUTPUT_DIR / "forecasting_cleaned.csv"
)


