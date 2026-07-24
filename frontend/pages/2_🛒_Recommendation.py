import pandas as pd
import streamlit as st

from api import recommend

st.title("🛒 Product Recommendation")

product_id = st.text_input(
    "Product ID"
)

top_n = st.slider(
    "Top N",
    1,
    10,
    5,
)

if st.button("Recommend"):

    if not product_id.strip():

        st.warning(
            "Please enter a Product ID."
        )

    else:

        response = recommend(
            {
                "product_id": product_id.strip(),
                "top_n": top_n,
            }
        )

        st.info(
            f"Recommendation Type : {response['recommendation_type']}"
        )

        st.write(
            f"Requested Product : {response['requested_product']}"
        )

        st.write(
            f"Recommendations Found : {response['recommendation_count']}"
        )

        st.dataframe(
            pd.DataFrame(
                response["recommendations"]
            ),
            use_container_width=True,
        )