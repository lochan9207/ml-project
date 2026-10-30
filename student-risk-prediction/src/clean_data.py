import pandas as pd

df = pd.read_csv("data/student-mat.csv", sep=";")
print("Shape before cleaning:", df.shape)

print("\nData types:")
print(df.dtypes)

# 1. Missing values
print("\n1. Missing values:", df.isnull().sum().sum())

# 2. Duplicates
print("2. Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# 3. Remove stray spaces in text columns
text_cols = df.select_dtypes(exclude="number").columns
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
print("3. Text columns trimmed:", len(text_cols))

# 4. Check text columns for typos or unexpected values
print("\n4. Unique values in text columns:")
for col in text_cols:
    print(f"   {col}: {sorted(df[col].unique())}")

# 5. Check value ranges
print("\n5. Value range checks:")
for col in ["G1", "G2", "G3"]:
    ok = df[col].between(0, 20).all()
    print(f"   {col} within 0-20: {ok} (min {df[col].min()}, max {df[col].max()})")
print(f"   age: {df['age'].min()} to {df['age'].max()}")
print(f"   absences: {df['absences'].min()} to {df['absences'].max()}")

# 6. Outliers in absences (IQR rule). We check them but keep them.
q1, q3 = df["absences"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
n_out = (df["absences"] > upper).sum()
print(f"\n6. Absences above {upper:.0f} (outliers): {n_out} students (kept)")

# 7. Students with a final grade of 0
print("7. Students with G3 = 0:", (df["G3"] == 0).sum(), "(kept)")

# Save the cleaned file
df.to_csv("data/student-mat-cleaned.csv", index=False)
print("\nShape after cleaning:", df.shape)
print("Saved: data/student-mat-cleaned.csv")


