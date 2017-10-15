import os
import pandas as pd
df = pd.read_csv("label_translation.csv")
new_labels = df["final_label"].unique()
new_path = '/home/jeanbaptiste/new_train'
old_path = '/home/jeanbaptiste/train'

for s in new_labels: 
	is_selected = df["final_label"] == s
	corresponding_kaggle = df[is_selected]["kaggle_label"]

	command = "mkdir {0}/{1}".format(new_path,s)
	print(command)
	os.system(command)

	for item in corresponding_kaggle: 
		command2 = "cp {0}/{1}/*.jpg {2}/{3}/".format(
			old_path,
			item,
			new_path,
			s
		)
		print(command2)
		os.system(command2)
