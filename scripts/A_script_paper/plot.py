# %%

import pandas as pd

df = pd.read_csv(
    "/home/arnault/traffic/data_orly/figures/paper/compare_profile/profile_end/mean_profile_no_filter_centroid_mean_profiles.csv"
)
# %%
import altair as alt

alt.Chart(df).mark_line().encode(
    alt.X("CAS"),
    alt.Y("altitude").title(None),
    alt.Color("generated"),
    alt.StrokeDash("generated"),
    alt.Row("typecode").title(None),
).properties(height=150)

# %%
