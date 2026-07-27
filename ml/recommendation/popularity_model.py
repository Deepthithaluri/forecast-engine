import pandas as pd

from ml.config import PROCESSED_DATA_PATH


POPULAR_PRODUCTS = None


def load_popular_products() -> None:
    """
    Load and cache the most frequently purchased products
    from the cleaned UCI Online Retail dataset.
    """

    global POPULAR_PRODUCTS

    df = pd.read_csv(
        PROCESSED_DATA_PATH / "cleaned_orders.csv"
    )

    # Count how many times each product appears
    POPULAR_PRODUCTS = (
        df.groupby(
            [
                "StockCode",
                "Description",
            ]
        )
        .size()
        .reset_index(name="purchase_count")
        .sort_values(
            by="purchase_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def get_popular_products(top_n: int = 10):
    """
    Return cached popular products.
    """

    if POPULAR_PRODUCTS is None:
        load_popular_products()

    return POPULAR_PRODUCTS.head(top_n)


if __name__ == "__main__":

    load_popular_products()

    print("=" * 60)
    print("Top 10 Popular Products")
    print("=" * 60)

    print(get_popular_products())