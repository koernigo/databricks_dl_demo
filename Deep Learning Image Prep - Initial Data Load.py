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

img_size = 299

def scale_image(image_bytes):
  image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
  # Scale image down
  image.thumbnail((img_size, img_size), Image.ANTIALIAS)
  x, y = image.size
  # Add border to make it square
  with_bg = Image.new('RGB', (img_size, img_size), (255, 255, 255))
  with_bg.paste(image, box=((img_size - x) // 2, (img_size - y) // 2))
  return with_bg.tobytes()

def file_to_label(path):
  # .../043.coin/043_0042.jpg -> 043.coin -> 043 -> 43
  return int(path.split("/")[-2].split(".")[-2])

scale_image_udf = udf(scale_image, BinaryType())
file_to_label_udf = udf(file_to_label, IntegerType())

# COMMAND ----------

import io
import numpy as np
from PIL import Image
from pyspark.sql.types import BinaryType, IntegerType

# COMMAND ----------

raw_image_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").load(caltech_256_path).repartition(64).limit(100)
image_df = raw_image_df.select(file_to_label_udf("path").alias("label"), scale_image_udf("content").alias("image")).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC The next step creates two Delta tables: One that stores all the raw images and one that stores all the labeled images

# COMMAND ----------

raw_image_df.write.format("delta").mode("overwrite").saveAsTable("raw_images")
image_df.write.format("delta").mode("overwrite").saveAsTable("labeled_images")
