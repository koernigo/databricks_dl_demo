# Databricks notebook source
import os
import re

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists dl_demo.image_data

# COMMAND ----------

# MAGIC %sh rm -rf /dbfs/tmp/dl_demo/images_data

# COMMAND ----------

# MAGIC %sh rm -rf /dbfs/tmp/chkpt/dl_demo/training/image_data2

# COMMAND ----------

# MAGIC %sh rm -rf /dbfs/tmp/chkpt/dl_demo/training/image_data

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE dl_demo.image_data 
# MAGIC (path STRING
# MAGIC , modificationTime TIMESTAMP
# MAGIC , length LONG
# MAGIC , content BINARY
# MAGIC , label INT
# MAGIC , load_date DATE
# MAGIC , predicted_label INT)
# MAGIC USING DELTA
# MAGIC LOCATION '/tmp/dl_demo/images_data'

# COMMAND ----------

# MAGIC %md
# MAGIC One thing that we want to do to make this more realistic is seperate our image files into two folders, a labeled folder that we can use for training, and a scoring folder for unlabeled images that need to be run through the model. To do this, we take one image from every image category and move it to a scoring folder `/tmp/unlabeled_images/`. 

# COMMAND ----------

paths = []
root = '/dbfs/tmp/256_ObjectCategories'
for root, dirs, files in os.walk(root):
  for f in files:
    if re.match('.*_0001.jpg', f):
      paths.append(os.path.join(root, f))


# COMMAND ----------

for path in paths:
  dbutils.fs.mv(path[5:], '/tmp/unlabeled_images/'+path[9:])

# COMMAND ----------


