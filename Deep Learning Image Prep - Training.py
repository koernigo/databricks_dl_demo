# Databricks notebook source
from pyspark.sql.functions import current_date, col

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation - Training
# MAGIC 
# MAGIC This loads the Caltech 256 images from .jpg files and extracts the label from the file name. The result is written into a Delta table. If new labeled images are added, autoloader will pick them up on subsequent runs. 

# COMMAND ----------

dbutils.widgets.text("image_path","/tmp/256_ObjectCategories/")
caltech_256_path = dbutils.widgets.get("image_path")

# COMMAND ----------

# MAGIC %sql
# MAGIC use dl_demo

# COMMAND ----------

# MAGIC %md
# MAGIC A pandas user-defined function (UDF)—also known as vectorized UDF—is a user-defined function that uses Apache Arrow to transfer data and pandas to work with the data. pandas UDFs allow vectorized operations that can increase performance up to 100x compared to row-at-a-time Python UDFs.
# MAGIC 
# MAGIC Here we create a Pandas UDF that parses the file name of the image and returns the label. 

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/tmp/256_ObjectCategories

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf /dbfs//tmp/chkpt/dl_demo/training/image_data

# COMMAND ----------

import numpy as np
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import pandas_udf

def file_to_label(path):
  # .../043.coin/043_0042.jpg -> 043.coin -> 043 -> 43
  
  return path.str.split(pat='/', expand=True).iloc[:,-2] \
             .str.split(pat=".", expand=True).iloc[:,-2] \
             .astype(int)

file_to_label_udf = pandas_udf(file_to_label, IntegerType())

# COMMAND ----------

#raw_image_df = spark.read.format("binaryFile") \
#                  .option("pathGlobFilter", "*.jpg") \
#                  .option("recursiveFileLookup", "true") \
      #             .load(caltech_256_path)



raw_image_df = spark.readStream.format("cloudFiles") \
              .option("cloudFiles.format", "binaryFile") \
              .option("recursiveFileLookup", "true") \
              .option("pathGlobFilter", "*.jpg") \
              .load(caltech_256_path) 


image_df = raw_image_df.withColumn("label",file_to_label_udf("path")) \
                       .withColumn("load_date", current_date())

# COMMAND ----------

image_df.writeStream \
  .format("delta") \
  .option("checkpointLocation", "/tmp/chkpt/dl_demo/training/image_data") \
  .trigger(once=True) \
  .option("mergeSchema", True) \
  .start("/tmp/dl_demo/images_data")
#display(image_df)

# COMMAND ----------

#image_df_filter.write.format("delta") \
#                     .mode("overwrite") \
#                     .option("mergeSchema", True) \
#                     .saveAsTable("image_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC optimize image_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from image_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from image_data where label is not null
