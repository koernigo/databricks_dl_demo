# Databricks notebook source
# MAGIC %md
# MAGIC To manually label a random selection of low scoring images, 

# COMMAND ----------

# MAGIC %sql
# MAGIC use dl_demo

# COMMAND ----------

new_images_tobe_labeled = spark.sql("select * from image_data where predicted_score < 95 and predicted_score is not null and label is null").sample(.1, 123)

# COMMAND ----------

paths = new_images_tobe_labeled.select("path").collect()

# COMMAND ----------

# MAGIC %md
# MAGIC Copy images to label manually to labeling folder

# COMMAND ----------

for path in paths:
  dbutils.fs.cp(path['path'][5:], '/tmp/manual_label/'+path['path'][9:])

# COMMAND ----------

# MAGIC %sh ls -R /dbfs/tmp/manual_label/

# COMMAND ----------

# MAGIC %md Create Input content Json objects

# COMMAND ----------

import json
#read new images to be processed from label table
new_images_tobe_labeled = spark.sql("select * from image_data where predicted_score < 95 and predicted_score is not null")

#Write to json
result = new_images_tobe_labeled \
  .select("path","predicted_label") \
  .write.mode("overwrite") \
  .option("multiline","false") \
  .json("/tmp/new_images.json")

json_images = new_images_tobe_labeled.toJSON().collect()

# COMMAND ----------

import json
input_content_str = json.dumps(json_images)

# COMMAND ----------

print(input_content_str)
