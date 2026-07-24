import pandas as pd

from ml.config import PROCESSED_DATA_PATH

POPULAR_PRODUCTS = None


def load_popular_products() -> None:
    """
    Load and cache popular products.
    """

    global POPULAR_PRODUCTS

    df = pd.read_csv(
        PROCESSED_DATA_PATH / "integrated_dataset.csv"
    )

    POPULAR_PRODUCTS = (
        df.groupby(
            [
                "product_id",
                "product_category_name_english",
            ]
        )
        .size()
        .reset_index(name="purchase_count")
        .sort_values(
            by="purchase_count",
            ascending=False,
        )
    )


def get_popular_products(
    top_n: int = 10,
):
    """
    Return cached popular products.
    """

    if POPULAR_PRODUCTS is None:
        load_popular_products()

    return POPULAR_PRODUCTS.head(top_n)


if __name__ == "__main__":

    load_popular_products()

    print(get_popular_products())