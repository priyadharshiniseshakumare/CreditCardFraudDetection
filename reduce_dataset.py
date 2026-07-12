import pandas as pd

# Read original test dataset
df = pd.read_csv("fraudTest.csv")

# Separate fraud and genuine transactions
fraud = df[df["is_fraud"] == 1]
genuine = df[df["is_fraud"] == 0]

# Take all fraud rows
# Take 5000 genuine rows
small_genuine = genuine.sample(5000, random_state=42)

# Combine both
small_df = pd.concat([fraud, small_genuine])

# Shuffle the rows
small_df = small_df.sample(frac=1, random_state=42)

# Save new dataset
small_df.to_csv("fraudTest_small.csv", index=False)

print("Dataset Created Successfully")
print("Rows:", len(small_df))