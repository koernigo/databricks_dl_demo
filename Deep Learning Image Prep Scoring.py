# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Preparation Image Scoring
# MAGIC 
# MAGIC This loads the new batch of images to be scored (for the demo, it's the same set of images)
# MAGIC This loads the Caltech 256 images from .jpg files, resizes them to 299x299, and extracts the label from the file name. The result is written to a Delta Table. We are going to look at this like these are new JPG images coming in, and that we are loading them daily to a table called `new_images`. 

# COMMAND ----------

# DBTITLE 1,Checkpoint location needs to be deleted, otherwise no new images will be processed if this is run more than once
# MAGIC %fs
# MAGIC rm -r /tmp/chkpt/dl_demo/scoring/image_data

# COMMAND ----------

from pyspark.sql.functions import current_date, col

# COMMAND ----------

# MAGIC %sql
# MAGIC use dl_demo

# COMMAND ----------

dbutils.widgets.text("image_path","/tmp/unlabeled_images/256_ObjectCategories/")
image_path = dbutils.widgets.get("image_path")

# COMMAND ----------

raw_image_df = spark.readStream.format("cloudFiles") \
              .option("cloudFiles.format", "binaryFile") \
              .option("recursiveFileLookup", "true") \
              .option("pathGlobFilter", "*.jpg") \
              .load(image_path) 

# COMMAND ----------

df_with_date = raw_image_df.withColumn("load_date", current_date())

# COMMAND ----------

df_with_date.writeStream \
  .format("delta") \
  .option("checkpointLocation", "/tmp/chkpt/dl_demo/scoring/image_data") \
  .trigger(once=True) \
  .option("mergeSchema", True) \
  .start("/tmp/dl_demo/images_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC select path from image_data where label is null

# COMMAND ----------


