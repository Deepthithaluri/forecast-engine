from pathlib import Path
import pandas as pd

DATA_DIR = Path("ml/data/raw/instacart")

orders = pd.read_csv(DATA_DIR / "orders.csv")
prior = pd.read_csv(DATA_DIR / "order_products__prior.csv")
train = pd.read_csv(DATA_DIR / "order_products__train.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
aisles = pd.read_csv(DATA_DIR / "aisles.csv")
departments = pd.read_csv(DATA_DIR / "departments.csv")

print("=" * 70)
print("INSTACART DATASET VERIFICATION")
print("=" * 70)

datasets = {
    "orders": orders,
    "order_products__prior": prior,
    "order_products__train": train,
    "products": products,
    "aisles": aisles,
    "departments": departments,
}

print("\nDATASET SHAPES")
print("-" * 70)

for name, df in datasets.items():
    print(f"{name:30} {df.shape}")

print("\nMISSING VALUES")
print("-" * 70)

for name, df in datasets.items():
    missing = df.isnull().sum().sum()
    print(f"{name:30} {missing}")

print("\nORDERS INFORMATION")
print("-" * 70)

print("Unique Users:", orders["user_id"].nunique())
print("Unique Orders:", orders["order_id"].nunique())

print("\nOrder Evaluation Set")
print(orders["eval_set"].value_counts())

print("\nDays Since Prior Order")
print(orders["days_since_prior_order"].describe())

print("\nPRODUCT INFORMATION")
print("-" * 70)

print("Unique Products:", products["product_id"].nunique())
print("Unique Departments:", departments["department_id"].nunique())
print("Unique Aisles:", aisles["aisle_id"].nunique())

print("\nPURCHASE INFORMATION")
print("-" * 70)

print("Prior Purchases:", len(prior))
print("Train Purchases:", len(train))

print("\nBasket Size Statistics")
basket_sizes = prior.groupby("order_id")["product_id"].count()

print(basket_sizes.describe())

print("\nAverage Basket Size:", basket_sizes.mean())
print("Maximum Basket Size:", basket_sizes.max())

print("\nOrders with >1 Product:", (basket_sizes > 1).sum())

print("\nTOP 20 PRODUCTS")
print("-" * 70)

top_products = (
    prior["product_id"]
    .value_counts()
    .head(20)
    .reset_index()
)

top_products.columns = ["product_id", "purchase_count"]

top_products = top_products.merge(
    products,
    on="product_id",
    how="left"
)

print(top_products[["product_name", "purchase_count"]])

print("\nSPARSITY CHECK")
print("-" * 70)

avg_products_per_user = (
    prior.merge(orders[["order_id", "user_id"]], on="order_id")
    .groupby("user_id")["product_id"]
    .count()
)

print("Average purchases per user:", avg_products_per_user.mean())

print("\nVERIFICATION COMPLETE")