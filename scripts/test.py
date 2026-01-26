import pandas as pd
df = pd.read_csv("../outputs/ppo_runs/seed42/eval/trajectory.csv")
print(df.columns)
print(df[["day","Dr","Dr_lo","Dr_hi","I"]].head())
