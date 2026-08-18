from pathlib import Path
import pandas as pd

DATA_DIR = Path("ml/data/raw/benroshan")

orders = pd.read_csv(DATA_DIR / "List of Orders.csv")
details = pd.read_csv(DATA_DIR / "Order Details.csv")

print("=" * 80)
print("BENROSHAN DATASET VERIFICATION")
print("=" * 80)

print("\nDATASET SHAPES")
print("-" * 80)
print(f"Orders Dataset : {orders.shape}")
print(f"Order Details  : {details.shape}")

print("\nCOLUMNS")
print("-" * 80)
print("Orders:", list(orders.columns))
print("Details:", list(details.columns))

print("\nMISSING VALUES")
print("-" * 80)
print("\nOrders")
print(orders.isnull().sum())

print("\nOrder Details")
print(details.isnull().sum())


orders["Order Date"] = pd.to_datetime(
    orders["Order Date"],
    dayfirst=True,
    errors="coerce"
)

print("\nDATE RANGE")
print("-" * 80)
print("Start :", orders["Order Date"].min())
print("End   :", orders["Order Date"].max())
print("Unique Dates :", orders["Order Date"].nunique())

print("\nORDERS")
print("-" * 80)
print("Total Orders :", orders["Order ID"].nunique())
print("Unique Customers :", orders["CustomerName"].nunique())

print("\nORDER DETAILS")
print("-" * 80)
print("Unique Products :", details["Category"].nunique())
print("Total Line Items :", len(details))


merged = pd.merge(
    orders,
    details,
    on="Order ID",
    how="inner"
)

print("\nMERGED DATA")
print("-" * 80)
print(merged.shape)

print("\nTOTAL SALES")
print("-" * 80)
print(f"₹ {merged['Amount'].sum():,.2f}")


daily_sales = (
    merged
    .groupby("Order Date")["Amount"]
    .sum()
    .reset_index()
)

print("\nFORECASTING STATISTICS")
print("-" * 80)
print("Daily Observations :", len(daily_sales))

print("\nDaily Sales Summary")
print(daily_sales["Amount"].describe())


full_dates = pd.date_range(
    daily_sales["Order Date"].min(),
    daily_sales["Order Date"].max(),
    freq="D"
)

missing_days = full_dates.difference(daily_sales["Order Date"])

print("\nMISSING DAYS")
print("-" * 80)
print("Number of Missing Days :", len(missing_days))


orders_per_day = (
    merged
    .groupby("Order Date")["Order ID"]
    .nunique()
)

print("\nORDERS PER DAY")
print("-" * 80)
print(orders_per_day.describe())


print("\nCATEGORY DISTRIBUTION")
print("-" * 80)
print(
    details["Category"]
    .value_counts()
)


if "State" in orders.columns:
    print("\nSTATE DISTRIBUTION")
    print("-" * 80)
    print(
        orders["State"]
        .value_counts()
    )


print("\nFINAL CHECKLIST")
print("-" * 80)

checks = {
    "Order IDs": orders["Order ID"].nunique() > 0,
    "Dates Present": orders["Order Date"].notna().all(),
    "Sales Amount": merged["Amount"].sum() > 0,
    "Daily Observations >= 300": len(daily_sales) >= 300,
    "Multiple Orders": orders["Order ID"].nunique() > 1000,
}

for item, result in checks.items():
    print(f"{item:<35} {'PASS' if result else 'FAIL'}")

print("\nVERIFICATION COMPLETE")