import os, csv
import pandas as pd


# DATA EXTRACTION

PATH = "/home/jeanbaptiste/train"
l = [(os.path.basename(path), files) for path, dirs, files in os.walk("/home/jeanbaptiste/train")]
df = pd.DataFrame(l, columns=['name','images'])
res = df.set_index('name')['images'].apply(pd.Series).stack()

res = res.reset_index()
res.columns = ["kaggle_label", "item", "image_name"]

# WRITE IN CSV

res.to_csv('image_list.csv', columns=['image_name','kaggle_label'], index=False)
