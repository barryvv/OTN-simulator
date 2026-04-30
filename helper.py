
import pandas as pd
import os














df = pd.DataFrame([])
for root, dirs, files in os.walk("outputs/testing B"):
    for d in dirs:
        print(str(d))
        for sub_root, sub_dirs, sub_files in os.walk(os.path.join(root, d)):
            for f in sub_files:
                if 'alarm_flows.csv' in f:
                    df_f = pd.read_csv(os.path.join(sub_root, f))
                    df = pd.concat([df, df_f], ignore_index=True)

df_output = pd.DataFrame([], columns=["s-time", "t-time", "s-board", "t-board", "af", "Label"])
df_output["s-time"] = df["s-time"]
df_output["t-time"] = df["t-time"]
df_output["s-board"] = df["s-board"]
df_output["t-board"] = df["t-board"]
df_output["af"] = df["af"]
df_output["Label"] = df["Label"]
df_output.to_csv("outputs/testing B/alarm_flows.csv", index=False)