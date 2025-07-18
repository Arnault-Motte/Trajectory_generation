# %%

import pandas as pd

df = pd.read_csv(
    "/home/arnault/traffic/data_orly/figures/paper/compare_profile/profile_end/mean_profile_no_filter_centroid_mean_profiles.csv"
)
# %%
import altair as alt
alt.data_transformers.enable("vegafusion")

alt.Chart(df).mark_line().encode(
    alt.X("CAS"),
    alt.Y("altitude").title(None),
    alt.Color("generated"),
    alt.Row("typecode").title(None),
    strokeDash= alt.condition(
        "datum.generated == 'flown' || datum.generated == 'generated'",
        alt.value([1,0]),
        alt.value([4,4]),
    ),
    opacity = alt.condition(
        'datum.generated == "flown" || datum.generated == "generated"',
        alt.value(1),
        alt.value(0.4),

    )

).properties(height=150)

# %%
