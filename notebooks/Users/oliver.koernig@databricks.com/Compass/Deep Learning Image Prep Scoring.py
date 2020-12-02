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
caltech_256_path = dbutils.widgets.get("image_path")
table_path=dbutils.widgets.get("table_path")

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

raw_image_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").load(caltech_256_path).repartition(64)

# COMMAND ----------

from pyspark.sql.functions import current_date


df_with_date = raw_image_df.withColumn("load_date", current_date())
df_with_date.write.partitionBy("load_date").format("delta").mode("append").option("mergeSchema", "true").saveAsTable("new_images")

# COMMAND ----------

# MAGIC %sql select count(*) from new_images where load_date = current_date()

# COMMAND ----------

