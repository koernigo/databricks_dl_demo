# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Preparation - Training
# MAGIC 
# MAGIC This loads the Caltech 256 images from .jpg files and extracts the label from the file name. The result is written into a Delta table. If new labeled images are added, autoloader will pick them up on subsequent runs. 

# COMMAND ----------

from pyspark.sql.functions import current_date, col, split, element_at, substring
import numpy as np
from pyspark.sql.types import IntegerType

# COMMAND ----------

dbutils.widgets.text("image_path","/tmp/256_ObjectCategories/")
caltech_256_path = dbutils.widgets.get("image_path")

# COMMAND ----------

# MAGIC %sql
# MAGIC use dl_demo

# COMMAND ----------

# DBTITLE 1,Create AutoLoader read stream and extract labels from file names
raw_image_df = spark.readStream.format("cloudFiles") \
              .option("cloudFiles.format", "binaryFile") \
              .option("recursiveFileLookup", "true") \
              .option("pathGlobFilter", "*.jpg") \
              .load(caltech_256_path)


image_df = raw_image_df.withColumn("label", substring(element_at(split(raw_image_df['path'], '/'), -2),1,3).cast(IntegerType())) \
                       .withColumn("load_date", current_date())

# COMMAND ----------

# DBTITLE 1,Write Images to the IMAGES_DATA Delta table
image_df.writeStream \
  .format("delta") \
  .option("checkpointLocation", "/tmp/chkpt/dl_demo/training/image_data") \
  .trigger(once=True) \
  .option("mergeSchema", True) \
  .start("/tmp/dl_demo/images_data")


# COMMAND ----------

# MAGIC %sql
# MAGIC optimize image_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from dl_demo.image_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select path, label from image_data where label is not null

# COMMAND ----------


