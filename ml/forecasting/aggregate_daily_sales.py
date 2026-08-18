from pathlib import Path
import logging
import pandas as pd


# Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# Paths

INPUT_FILE = Path("ml/data/processed/forecasting_cleaned.csv")
OUTPUT_DIR = Path("ml/data/processed")
OUTPUT_FILE = OUTPUT_DIR / "daily_sales.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Validation

REQUIRED_COLUMNS = {
    "Order Date",
    "Amount"
}


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate required columns and missing values.
    """

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["Order Date"].isna().any():
        raise ValueError(
            "Order Date contains missing values."
        )

    if df["Amount"].isna().any():
        raise ValueError(
            "Amount contains missing values."
        )


# Main

def main():

    logger.info("Loading cleaned forecasting dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    logger.info("Loaded %d rows", len(df))


    # Convert Date

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    validate_input(df)

    logger.info("Input validation passed.")



    logger.info("Aggregating daily sales...")

    daily_sales = (
        df.groupby("Order Date", as_index=False)
          .agg(
              daily_sales=("Amount", "sum"),
              total_orders=("Order ID", "nunique"),
              total_quantity=("Quantity", "sum"),
              total_profit=("Profit", "sum")
          )
          .sort_values("Order Date")
    )

  

    logger.info("Generating continuous date range...")

    full_dates = pd.DataFrame({
        "Order Date": pd.date_range(
            start=daily_sales["Order Date"].min(),
            end=daily_sales["Order Date"].max(),
            freq="D"
        )
    })

    daily_sales = full_dates.merge(
        daily_sales,
        on="Order Date",
        how="left"
    )

    numeric_columns = [
        "daily_sales",
        "total_orders",
        "total_quantity",
        "total_profit"
    ]

    daily_sales[numeric_columns] = (
        daily_sales[numeric_columns]
        .fillna(0)
    )

  

    logger.info("Adding calendar features...")

    daily_sales["year"] = daily_sales["Order Date"].dt.year
    daily_sales["month"] = daily_sales["Order Date"].dt.month
    daily_sales["day"] = daily_sales["Order Date"].dt.day
    daily_sales["day_of_week"] = daily_sales["Order Date"].dt.dayofweek

    daily_sales["is_weekend"] = (
        daily_sales["day_of_week"] >= 5
    ).astype(int)

   

    daily_sales.to_csv(
        OUTPUT_FILE,
        index=False
    )

    logger.info("Saved: %s", OUTPUT_FILE)


    print("\n" + "=" * 70)
    print("DAILY SALES VALIDATION REPORT")
    print("=" * 70)

    print(f"Start Date          : {daily_sales['Order Date'].min().date()}")
    print(f"End Date            : {daily_sales['Order Date'].max().date()}")
    print(f"Total Days          : {len(daily_sales)}")

    zero_days = (daily_sales["daily_sales"] == 0).sum()

    print(f"Zero Sales Days     : {zero_days}")

    print(f"Total Sales         : ₹{daily_sales['daily_sales'].sum():,.2f}")
    print(f"Average Daily Sales : ₹{daily_sales['daily_sales'].mean():,.2f}")
    print(f"Maximum Daily Sales : ₹{daily_sales['daily_sales'].max():,.2f}")
    print(f"Minimum Daily Sales : ₹{daily_sales['daily_sales'].min():,.2f}")

if __name__ == "__main__":
    main()