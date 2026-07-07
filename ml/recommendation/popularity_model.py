import pandas as pd

from ml.config import PROCESSED_DATA_PATH


def get_popular_products(top_n=10):
    df = pd.read_csv(
        PROCESSED_DATA_PATH / "integrated_dataset.csv"
    )

    popular_products = (
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

    return popular_products.head(top_n)


if __name__ == "__main__":
    print(get_popular_products())