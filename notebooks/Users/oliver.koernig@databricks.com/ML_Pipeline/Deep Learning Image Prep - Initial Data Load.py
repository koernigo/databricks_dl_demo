# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Preparation - Initial Data Load
# MAGIC 
# MAGIC This loads the Caltech 256 images from .jpg files, resizes them to 299x299, and extracts the label from the file name. The result is written to Parquet and Delta tables. This only needs to be run once.

# COMMAND ----------

dbutils.widgets.text("table_path","/ml/images/tables/")
table_path=dbutils.widgets.get("table_path")
dbutils.widgets.text("image_path","/tmp/ok/images/")
caltech_256_path = dbutils.widgets.get("image_path")
table_path=dbutils.widgets.get("table_path")

# COMMAND ----------

# MAGIC %run /Projects/ashley.trainor@databricks.com/databricks_dl_demo/notebooks/Users/oliver.koernig@databricks.com/ML_Pipeline/Functions

# COMMAND ----------

import io
import numpy as np
from PIL import Image
from pyspark.sql.types import BinaryType, IntegerType

# COMMAND ----------

raw_image_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").load(caltech_256_path).repartition(64)

image_df = raw_image_df.select(file_to_label_udf("path").alias("label"), scale_image_udf("content").alias("image")).cache()

# Go ahead and make a 90%/10% train/test split
(train_image_df, test_image_df) = image_df.randomSplit([0.9, 0.1], seed=42)

table_path_base = table_path+"pq/"
dbutils.fs.rm(table_path_base, True)

# COMMAND ----------

# MAGIC %md
# MAGIC The next step copies the images into dbfs/ml. dbfs/ml uses a faster file access library, so it will improve the time it takes to load the images for training

# COMMAND ----------

# parquet.block.size is for Petastorm, later
train_image_df.write.format("parquet").option("parquet.block.size", 1024 * 1024).save(table_path_base + "train")
test_image_df.write.format("parquet").option("parquet.block.size", 1024 * 1024).save(table_path_base + "test")

# COMMAND ----------

# MAGIC %md
# MAGIC The next step creates two Delta tables: One that stores all the raw images and one that stores all the labeled images

# COMMAND ----------

raw_image_df.write.format("delta").mode("overwrite").saveAsTable("raw_images")
image_df.write.format("delta").mode("overwrite").saveAsTable("labeled_images")

# COMMAND ----------

