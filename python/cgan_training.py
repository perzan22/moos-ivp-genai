import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# reading manifest file with runs conditions
manifest_path = "dataset/manifest.csv"
df = pd.read_csv(manifest_path)

print(df.head())

image_paths = df["traj_img"].tolist()
