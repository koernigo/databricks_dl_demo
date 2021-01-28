# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Preparation or Image Scoring
# MAGIC 
# MAGIC This loads the new batch of images to be scored (for the demo, it's the same set of images)
# MAGIC This loads the Caltech 256 images from .jpg files, resizes them to 299x299, and extracts the label from the file name. The result is written to a Delta Table. We are going to look at this like these are new JPG images coming in, and that we are loading them daily to a table called `new_images`. 

# COMMAND ----------

dbutils.widgets.text("table_path","/ml/images/tables/")
table_path=dbutils.widgets.get("table_path")
dbutils.widgets.text("image_path","/mnt/poc/images/caltech_256/")
image_path = dbutils.widgets.get("image_path")
table_path=dbutils.widgets.get("table_path")

# COMMAND ----------

# MAGIC %run /Projects/ashley.trainor@databricks.com/databricks_dl_demo/notebooks/Users/oliver.koernig@databricks.com/ML_Pipeline/Functions

# COMMAND ----------

raw_image_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").load(image_path).repartition(64)

# COMMAND ----------

from pyspark.sql.functions import current_date


df_with_date = raw_image_df.withColumn("load_date", current_date())
df_with_date.write.partitionBy("load_date").format("delta").mode("append").option("mergeSchema", "true").saveAsTable("new_images")

# COMMAND ----------

# MAGIC %sql select count(*) from new_images where load_date = current_date()