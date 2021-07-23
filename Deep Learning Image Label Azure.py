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
