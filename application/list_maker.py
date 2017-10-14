import os, csv
import pandas as pd


#-----------------------FILE 1: image name / kaggle label 

# DATA EXTRACTION

PATH = "/home/jeanbaptiste/train"
l = [
	(os.path.basename(path), files) 
	for path, dirs, files in os.walk("/home/jeanbaptiste/train")
]
df = pd.DataFrame(l, columns=['name','images'])
res = df.set_index('name')['images'].apply(pd.Series).stack()

res = res.reset_index()
res.columns = ["kaggle_label", "item", "image_name"]

# WRITE IN CSV

res.to_csv(
	'image_list.csv', 
	columns=['image_name','kaggle_label'],
	index=False
)

#---------------------- FILE 2: image name/ clustered label

#  READ TRANSLATION FILE

trans_file = 'label_translation.csv'
trans_df = pd.read_csv(trans_file)

df=pd.merge(res, trans_df, on='kaggle_label')

# WRITE IN CSV

df.to_csv(
	'newlabels_image_list.csv',
	columns=['image_name','final_label'],
	index=False
)
