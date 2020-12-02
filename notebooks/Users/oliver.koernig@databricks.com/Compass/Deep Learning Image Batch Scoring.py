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

schema = StructType([
StructField("path", StringType(),True),
StructField("modificationTime", TimestampType(),True),
StructField("length",LongType(),True),
StructField("load_date",DateType(),True),
StructField("label_0",FloatType(),True),
StructField("label_1",FloatType(),True),
StructField("label_2",FloatType(),True),
StructField("label_3",FloatType(),True),
StructField("label_4",FloatType(),True),
StructField("label_5",FloatType(),True),
StructField("label_6",FloatType(),True),
StructField("label_7",FloatType(),True),
StructField("label_8",FloatType(),True),
StructField("label_9",FloatType(),True),
StructField("label_10",FloatType(),True),
StructField("label_11",FloatType(),True),
StructField("label_12",FloatType(),True),
StructField("label_13",FloatType(),True),
StructField("label_14",FloatType(),True),
StructField("label_15",FloatType(),True),
StructField("label_16",FloatType(),True),
StructField("label_17",FloatType(),True),
StructField("label_18",FloatType(),True),
StructField("label_19",FloatType(),True),
StructField("label_20",FloatType(),True),
StructField("label_21",FloatType(),True),
StructField("label_22",FloatType(),True),
StructField("label_23",FloatType(),True),
StructField("label_24",FloatType(),True),
StructField("label_25",FloatType(),True),
StructField("label_26",FloatType(),True),
StructField("label_27",FloatType(),True),
StructField("label_28",FloatType(),True),
StructField("label_29",FloatType(),True),
StructField("label_30",FloatType(),True),
StructField("label_31",FloatType(),True),
StructField("label_32",FloatType(),True),
StructField("label_33",FloatType(),True),
StructField("label_34",FloatType(),True),
StructField("label_35",FloatType(),True),
StructField("label_36",FloatType(),True),
StructField("label_37",FloatType(),True),
StructField("label_38",FloatType(),True),
StructField("label_39",FloatType(),True),
StructField("label_40",FloatType(),True),
StructField("label_41",FloatType(),True),
StructField("label_42",FloatType(),True),
StructField("label_43",FloatType(),True),
StructField("label_44",FloatType(),True),
StructField("label_45",FloatType(),True),
StructField("label_46",FloatType(),True),
StructField("label_47",FloatType(),True),
StructField("label_48",FloatType(),True),
StructField("label_49",FloatType(),True),
StructField("label_50",FloatType(),True),
StructField("label_51",FloatType(),True),
StructField("label_52",FloatType(),True),
StructField("label_53",FloatType(),True),
StructField("label_54",FloatType(),True),
StructField("label_55",FloatType(),True),
StructField("label_56",FloatType(),True),
StructField("label_57",FloatType(),True),
StructField("label_58",FloatType(),True),
StructField("label_59",FloatType(),True),
StructField("label_60",FloatType(),True),
StructField("label_61",FloatType(),True),
StructField("label_62",FloatType(),True),
StructField("label_63",FloatType(),True),
StructField("label_64",FloatType(),True),
StructField("label_65",FloatType(),True),
StructField("label_66",FloatType(),True),
StructField("label_67",FloatType(),True),
StructField("label_68",FloatType(),True),
StructField("label_69",FloatType(),True),
StructField("label_70",FloatType(),True),
StructField("label_71",FloatType(),True),
StructField("label_72",FloatType(),True),
StructField("label_73",FloatType(),True),
StructField("label_74",FloatType(),True),
StructField("label_75",FloatType(),True),
StructField("label_76",FloatType(),True),
StructField("label_77",FloatType(),True),
StructField("label_78",FloatType(),True),
StructField("label_79",FloatType(),True),
StructField("label_80",FloatType(),True),
StructField("label_81",FloatType(),True),
StructField("label_82",FloatType(),True),
StructField("label_83",FloatType(),True),
StructField("label_84",FloatType(),True),
StructField("label_85",FloatType(),True),
StructField("label_86",FloatType(),True),
StructField("label_87",FloatType(),True),
StructField("label_88",FloatType(),True),
StructField("label_89",FloatType(),True),
StructField("label_90",FloatType(),True),
StructField("label_91",FloatType(),True),
StructField("label_92",FloatType(),True),
StructField("label_93",FloatType(),True),
StructField("label_94",FloatType(),True),
StructField("label_95",FloatType(),True),
StructField("label_96",FloatType(),True),
StructField("label_97",FloatType(),True),
StructField("label_98",FloatType(),True),
StructField("label_99",FloatType(),True),
StructField("label_100",FloatType(),True),
StructField("label_101",FloatType(),True),
StructField("label_102",FloatType(),True),
StructField("label_103",FloatType(),True),
StructField("label_104",FloatType(),True),
StructField("label_105",FloatType(),True),
StructField("label_106",FloatType(),True),
StructField("label_107",FloatType(),True),
StructField("label_108",FloatType(),True),
StructField("label_109",FloatType(),True),
StructField("label_110",FloatType(),True),
StructField("label_111",FloatType(),True),
StructField("label_112",FloatType(),True),
StructField("label_113",FloatType(),True),
StructField("label_114",FloatType(),True),
StructField("label_115",FloatType(),True),
StructField("label_116",FloatType(),True),
StructField("label_117",FloatType(),True),
StructField("label_118",FloatType(),True),
StructField("label_119",FloatType(),True),
StructField("label_120",FloatType(),True),
StructField("label_121",FloatType(),True),
StructField("label_122",FloatType(),True),
StructField("label_123",FloatType(),True),
StructField("label_124",FloatType(),True),
StructField("label_125",FloatType(),True),
StructField("label_126",FloatType(),True),
StructField("label_127",FloatType(),True),
StructField("label_128",FloatType(),True),
StructField("label_129",FloatType(),True),
StructField("label_130",FloatType(),True),
StructField("label_131",FloatType(),True),
StructField("label_132",FloatType(),True),
StructField("label_133",FloatType(),True),
StructField("label_134",FloatType(),True),
StructField("label_135",FloatType(),True),
StructField("label_136",FloatType(),True),
StructField("label_137",FloatType(),True),
StructField("label_138",FloatType(),True),
StructField("label_139",FloatType(),True),
StructField("label_140",FloatType(),True),
StructField("label_141",FloatType(),True),
StructField("label_142",FloatType(),True),
StructField("label_143",FloatType(),True),
StructField("label_144",FloatType(),True),
StructField("label_145",FloatType(),True),
StructField("label_146",FloatType(),True),
StructField("label_147",FloatType(),True),
StructField("label_148",FloatType(),True),
StructField("label_149",FloatType(),True),
StructField("label_150",FloatType(),True),
StructField("label_151",FloatType(),True),
StructField("label_152",FloatType(),True),
StructField("label_153",FloatType(),True),
StructField("label_154",FloatType(),True),
StructField("label_155",FloatType(),True),
StructField("label_156",FloatType(),True),
StructField("label_157",FloatType(),True),
StructField("label_158",FloatType(),True),
StructField("label_159",FloatType(),True),
StructField("label_160",FloatType(),True),
StructField("label_161",FloatType(),True),
StructField("label_162",FloatType(),True),
StructField("label_163",FloatType(),True),
StructField("label_164",FloatType(),True),
StructField("label_165",FloatType(),True),
StructField("label_166",FloatType(),True),
StructField("label_167",FloatType(),True),
StructField("label_168",FloatType(),True),
StructField("label_169",FloatType(),True),
StructField("label_170",FloatType(),True),
StructField("label_171",FloatType(),True),
StructField("label_172",FloatType(),True),
StructField("label_173",FloatType(),True),
StructField("label_174",FloatType(),True),
StructField("label_175",FloatType(),True),
StructField("label_176",FloatType(),True),
StructField("label_177",FloatType(),True),
StructField("label_178",FloatType(),True),
StructField("label_179",FloatType(),True),
StructField("label_180",FloatType(),True),
StructField("label_181",FloatType(),True),
StructField("label_182",FloatType(),True),
StructField("label_183",FloatType(),True),
StructField("label_184",FloatType(),True),
StructField("label_185",FloatType(),True),
StructField("label_186",FloatType(),True),
StructField("label_187",FloatType(),True),
StructField("label_188",FloatType(),True),
StructField("label_189",FloatType(),True),
StructField("label_190",FloatType(),True),
StructField("label_191",FloatType(),True),
StructField("label_192",FloatType(),True),
StructField("label_193",FloatType(),True),
StructField("label_194",FloatType(),True),
StructField("label_195",FloatType(),True),
StructField("label_196",FloatType(),True),
StructField("label_197",FloatType(),True),
StructField("label_198",FloatType(),True),
StructField("label_199",FloatType(),True),
StructField("label_200",FloatType(),True),
StructField("label_201",FloatType(),True),
StructField("label_202",FloatType(),True),
StructField("label_203",FloatType(),True),
StructField("label_204",FloatType(),True),
StructField("label_205",FloatType(),True),
StructField("label_206",FloatType(),True),
StructField("label_207",FloatType(),True),
StructField("label_208",FloatType(),True),
StructField("label_209",FloatType(),True),
StructField("label_210",FloatType(),True),
StructField("label_211",FloatType(),True),
StructField("label_212",FloatType(),True),
StructField("label_213",FloatType(),True),
StructField("label_214",FloatType(),True),
StructField("label_215",FloatType(),True),
StructField("label_216",FloatType(),True),
StructField("label_217",FloatType(),True),
StructField("label_218",FloatType(),True),
StructField("label_219",FloatType(),True),
StructField("label_220",FloatType(),True),
StructField("label_221",FloatType(),True),
StructField("label_222",FloatType(),True),
StructField("label_223",FloatType(),True),
StructField("label_224",FloatType(),True),
StructField("label_225",FloatType(),True),
StructField("label_226",FloatType(),True),
StructField("label_227",FloatType(),True),
StructField("label_228",FloatType(),True),
StructField("label_229",FloatType(),True),
StructField("label_230",FloatType(),True),
StructField("label_231",FloatType(),True),
StructField("label_232",FloatType(),True),
StructField("label_233",FloatType(),True),
StructField("label_234",FloatType(),True),
StructField("label_235",FloatType(),True),
StructField("label_236",FloatType(),True),
StructField("label_237",FloatType(),True),
StructField("label_238",FloatType(),True),
StructField("label_239",FloatType(),True),
StructField("label_240",FloatType(),True),
StructField("label_241",FloatType(),True),
StructField("label_242",FloatType(),True),
StructField("label_243",FloatType(),True),
StructField("label_244",FloatType(),True),
StructField("label_245",FloatType(),True),
StructField("label_246",FloatType(),True),
StructField("label_247",FloatType(),True),
StructField("label_248",FloatType(),True),
StructField("label_249",FloatType(),True),
StructField("label_250",FloatType(),True),
StructField("label_251",FloatType(),True),
StructField("label_252",FloatType(),True),
StructField("label_253",FloatType(),True),
StructField("label_254",FloatType(),True),
StructField("label_255",FloatType(),True),
StructField("label_256",FloatType(),True),
StructField("predicted_score", FloatType(), True),
StructField("predicted_label", LongType(), True)])

# COMMAND ----------

# MAGIC %md
# MAGIC Now we write a Pandas UDF that loads the Production version of the MLflow Keras model, and then scores the images that we loaded from the Delta Table earlier in the notebook. 

# COMMAND ----------

import pandas as pd

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

