# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Preparation - Initial Data Load
# MAGIC 
# MAGIC This loads the Caltech 256 images from .jpg files, resizes them to 299x299, and extracts the label from the file name. The result is written to Parquet and Delta tables. This only needs to be run once.

# COMMAND ----------

dbutils.widgets.text("table_path","/ml/images/tables/")
table_path=dbutils.widgets.get("table_path")
dbutils.widgets.text("image_path","/tmp/256_ObjectCategories/")
caltech_256_path = dbutils.widgets.get("image_path")
table_path=dbutils.widgets.get("table_path")

# COMMAND ----------

# MAGIC %sql
# MAGIC use dl_demo

# COMMAND ----------

import io
import numpy as np
from PIL import Image
from pyspark.sql.types import BinaryType, IntegerType

def file_to_label(path):
  # .../043.coin/043_0042.jpg -> 043.coin -> 043 -> 43
  return int(path.split("/")[-2].split(".")[-2])

file_to_label_udf = udf(file_to_label, IntegerType())

# COMMAND ----------

import io
import numpy as np
from PIL import Image
from pyspark.sql.types import BinaryType, IntegerType

# COMMAND ----------

raw_image_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").load(caltech_256_path).repartition(64).limit(100)
image_df = raw_image_df.withColumn("label",file_to_label_udf("path")).cache()

# COMMAND ----------

image_df.write.format("delta").mode("overwrite").saveAsTable("labeled_images")

# COMMAND ----------


