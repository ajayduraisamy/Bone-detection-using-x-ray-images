import os
import pandas as pd

# Paths
csv_file = "train.csv"    
images_dir = "images"      
output_csv = "boneage.csv"

# Load CSV
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip().str.lower()

# Convert female column
def get_gender(female):
    return "female" if female == 1 else "male"

# Convert months → years
def months_to_years(months):
    return round(months / 12, 1)

# Define age category
def get_age_category(years):
    if years <= 12:
        return "child"
    elif years <= 19:
        return "teen"
    elif years <= 59:
        return "adult"
    else:
        return "senior"

# Add new columns
df["image_path"] = df["pid"].apply(lambda x: os.path.join(images_dir, f"{x}.png"))  
df["gender"] = df["female"].apply(get_gender)
df["age_years"] = df["bone_age"].apply(months_to_years)
df["age_category"] = df["age_years"].apply(get_age_category)

# Save new CSV
df.to_csv(output_csv, index=False)

print(" CSV updated with image_path, gender, age_years, age_category →", output_csv)
