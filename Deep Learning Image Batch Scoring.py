# Databricks notebook source
# MAGIC %md
# MAGIC Deep Learning Image Batch Scoring
# MAGIC 
# MAGIC This is a daily job that scores all images in the new_images table and appens the detailed into the image_label_results table

# COMMAND ----------

# Including MLflow
import mlflow
import os
print("MLflow Version: %s" % mlflow.__version__)

# COMMAND ----------

import io
import numpy as np
from PIL import Image
from pyspark.sql.types import BinaryType, IntegerType
from tensorflow.keras.applications.xception import preprocess_input
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import *
import mlflow
import mlflow.pyfunc

# COMMAND ----------

model_name = "deep_learning_demo"
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are reading in the Delta table with the new images from the last step as a Spark DataFrame.

# COMMAND ----------

df = spark.table("new_images") 

# COMMAND ----------

# MAGIC %md
# MAGIC Now we write a Pandas UDF that loads the Production version of the MLflow Keras model, and then scores the images that we loaded from the Delta Table earlier in the notebook. 

# COMMAND ----------

model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{'Production'}")
num_labels = model.output_shape[1]

schema_list = [
StructField("path", StringType(),True),
StructField("modificationTime", TimestampType(),True),
StructField("length",LongType(),True),
StructField("load_date",DateType(),True)]

label_list = [StructField("label_{}".format(i), FloatType(), True) for i in range(num_labels)]

pred_list = [StructField("predicted_score", FloatType(), True),
StructField("predicted_label", LongType(), True)]

schema = StructType(schema_list + label_list + pred_list)

# COMMAND ----------

img_size = 299
from tensorflow import keras

# COMMAND ----------

def scale_image(image_bytes):
  image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
  # Scale image down
  image.thumbnail((img_size, img_size), Image.ANTIALIAS)
  x, y = image.size
  # Add border to make it square
  with_bg = Image.new('RGB', (img_size, img_size), (255, 255, 255))
  with_bg.paste(image, box=((img_size - x) // 2, (img_size - y) // 2))
  return with_bg.tobytes()

scale_image_udf = udf(scale_image, BinaryType())

# COMMAND ----------

import pandas as pd

img_size = 299

def predict_match_udf(image_dfs):
  #This loads the latest image scoring production model from the model registry
  model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{'Production'}")
  for image_df1 in image_dfs:
    X_raw = image_df1["image"].values
    X = np.array([preprocess_input(np.frombuffer(X_raw[i], dtype=np.uint8).reshape((img_size,img_size,3))) for i in range(len(X_raw))])
    
    predictions = model.predict(X)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_scores = np.amax(predictions * 100, axis=1)
    
    preds_pd = pd.DataFrame(predictions)
    prefix_df = preds_pd.add_prefix("label_")
    
    final_df = pd.concat([image_df1, prefix_df.reindex(image_df1.index)], axis=1)
    
    final_df["predicted_score"] = predicted_scores
    final_df["predicted_label"] = predicted_labels
    final_df.drop(columns=["content", "image"], inplace=True)
    
    yield pd.DataFrame(final_df)
 

# COMMAND ----------

image_df = df.withColumn("image", scale_image_udf("content"))
preds = image_df.mapInPandas(predict_match_udf, schema=schema)

# COMMAND ----------

display(preds)

# COMMAND ----------

# MAGIC %md Here we add a date column that we will use to partition the output of the model in another Delta Table.

# COMMAND ----------

preds.write.format("delta") \
              .partitionBy("load_date") \
              .mode("append") \
              .option("mergeSchema", "true") \
              .saveAsTable("image_label_results")

# COMMAND ----------

# MAGIC %md
# MAGIC The table below is an output of our model scoring. It contains the data from the original delta table with metadata on the raw images, the prediction for each label, as well as the predicted label and its score. 

# COMMAND ----------

# MAGIC %sql
# MAGIC --- Only need to run this for the first time, it will fail otherwise
# MAGIC ---alter table image_label_results ADD COLUMNS (manual_label FLOAT)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from image_label_results

# COMMAND ----------

# MAGIC %sql
# MAGIC --drop table if exists image_label_results

# COMMAND ----------

