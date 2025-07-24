import pickle
from pathlib import Path

import pandas as pd
import yaml

directory = Path(
    "/home/arnault/traffic/scripts/A_script_paper/model_spec/aircrafts_spec"
)
dict_spec = {}


engines = pd.read_csv(
    "/home/arnault/traffic/scripts/A_script_paper/model_spec/engines.csv"
)

for file_path in directory.glob("*.yml"):
    filename = file_path.name
    typecode = filename.split(".")[0].upper()
    print(typecode)
    dict_spec[typecode] = {}
    print(f"Filename: {filename}")
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        dict_spec[typecode]["mtow"] = data["mtow"]
        dict_spec[typecode]["wing_area"] = data["wing"]["area"]
        dict_spec[typecode]["wing_span"] = data["wing"]["span"]
        engine = data["engine"]["default"]
        engine_num = data["engine"]["number"]

        thrust_engine = float(
            engines[engines["name"].str.contains(engine)]["max_thrust"].iloc[0]
        )
        dict_spec[typecode]["thrust"] = thrust_engine * engine_num

print(dict_spec)
df = pd.DataFrame.from_dict(dict_spec)
print(df)
print(df["A320"]["mtow"])
print(df.columns)
df_normalized = df.sub(df.min(axis=1), axis=0).div(df.max(axis=1) - df.min(axis=1), axis=0)
print(df_normalized)

dic_normalize = df_normalized.to_dict()
print(dic_normalize)

with open("/home/arnault/traffic/scripts/A_script_paper/model_spec/dic_spec_norm.pkl", "wb") as handle:
    pickle.dump(dic_normalize, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("/home/arnault/traffic/scripts/A_script_paper/model_spec/dic_spec.pkl", "wb") as handle:
    pickle.dump(dict_spec, handle, protocol=pickle.HIGHEST_PROTOCOL)
