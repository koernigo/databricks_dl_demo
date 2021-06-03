# Databricks notebook source
# MAGIC %md
# MAGIC Imports

# COMMAND ----------

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow.data.experimental import unbatch
from tensorflow.io import decode_raw
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec
from petastorm.tf_utils import make_petastorm_dataset

from pyspark.sql.functions import col

import math
import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tempfile
from sparkdl import HorovodRunner
import random
import io
from PIL import Image

import horovod.tensorflow.keras as hvd
from sparkdl import HorovodRunner

# COMMAND ----------

# MAGIC %sql
# MAGIC use dl_demo;

# COMMAND ----------

# MAGIC %md
# MAGIC Next we build the model

# COMMAND ----------

IMG_SHAPE = (299, 299, 3)
img_size = IMG_SHAPE[0]

def build_model(dropout=None):
  model = Sequential()
  xception = Xception(include_top=False, input_shape=IMG_SHAPE, pooling='avg')
  for layer in xception.layers:
    layer.trainable = False
  model.add(xception)
  if dropout:
    model.add(Dropout(dropout))
  model.add(Dense(257, activation='softmax'))
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC now we load in the data

# COMMAND ----------

full_data = spark.table("labeled_images").select("content", "label").limit(1000)
df_train, df_val = full_data.randomSplit([0.9, 0.1], seed=12345)

num_classes = full_data.select("label").distinct().count()

# Make sure the number of partitions is at least the number of workers which is required for distributed training.
df_train = df_train.repartition(2)
df_val = df_val.repartition(2)

# COMMAND ----------

# MAGIC %md
# MAGIC  Use `/dbfs/ml` and Petastorm for Efficient Data Access
# MAGIC  
# MAGIC Petastorm is an open source data access library. This library enables single-node or distributed training and evaluation of deep learning models directly from datasets in Apache Parquet format and datasets that are already loaded as Apache Spark DataFrames. Petastorm supports popular Python-based machine learning (ML) frameworks such as Tensorflow, PyTorch, and PySpark. For more information about Petastorm, refer to the Petastorm GitHub page and Petastorm API documentation.

# COMMAND ----------

# Set a cache directory on DBFS FUSE for intermediate data.
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

# COMMAND ----------

def preprocess(content):
  """
  Preprocess an image file bytes for model
  """
  image = Image.open(io.BytesIO(content)).convert('RGB')
  # Scale image down
  image.thumbnail((img_size, img_size), Image.ANTIALIAS)
  x, y = image.size
  # Add border to make it square
  with_bg = Image.new('RGB', (img_size, img_size), (255, 255, 255))
  with_bg.paste(image, box=((img_size - x) // 2, (img_size - y) // 2))
  image_array = keras.preprocessing.image.img_to_array(with_bg)
  
  return preprocess_input(image_array)

def transform_row(pd_batch):
  """
  The input and output of this function are pandas dataframes.
  """
  pd_batch['features'] = pd_batch['content'].map(lambda x: preprocess(x))
  pd_batch = pd_batch.drop(labels='content', axis=1)
  return pd_batch

# The output shape of the `TransformSpec` is not automatically known by petastorm, 
# so you need to specify the shape for new columns in `edit_fields` and specify the order of 
# the output columns in `selected_fields`.
transform_spec_fn = TransformSpec(
  transform_row, 
  edit_fields=[('features', np.float32, IMG_SHAPE, False)], 
  selected_fields=['features', 'label']
)

# COMMAND ----------

# DBTITLE 1,Use MLflow for tracking and model registration
import mlflow

#import mlflow.keras
import mlflow.tensorflow
import os
import tempfile
print("MLflow Version: %s" % mlflow.__version__)
mlflow.tensorflow.autolog()

# COMMAND ----------

# Save Horovod timeline for later analysis
output_base = "/tmp/keras_horovodrunner_mlflow/"
dbutils.fs.rm(output_base, recurse=True)
dbutils.fs.mkdirs(output_base)
timeline_root = output_base + "hvd-demo_timeline.json"
timeline_path = "/dbfs" + timeline_root
os.environ['HOROVOD_TIMELINE'] = timeline_path

output_base = "dbfs:/FileStore/hvd-demo/"
output_dir = output_base + str(random.randint(0, 1000000))
dbutils.fs.mkdirs(output_dir)
output_dir = output_dir.replace("dbfs:", "/dbfs")
print(output_dir)

# Print out the `HorovodRunner` API signature for reference.
help(HorovodRunner)

# COMMAND ----------

databricks_host = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())["extraContext"]["api_url"]
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

checkpoint_path = "/tmp/ok/images/checkpoint"
dbutils.fs.rm("file:" + checkpoint_path, recurse=True)
dbutils.fs.mkdirs("file:" + checkpoint_path)

# COMMAND ----------

BATCH_SIZE = 32
num_gpus = 4
epochs = 12

# COMMAND ----------

def train_hvd():
  hvd.init()
  
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
  
  # Horovod: adjust number of epochs based on number of GPUs.
  print(hvd.size())
  hvd_epochs = int(math.ceil(epochs / hvd.size()))

  #pq.EXCLUDED_PARQUET_PATHS.update(underscore_files
  mlflow.mlflow.set_tracking_uri('databricks')
  os.environ['DATABRICKS_HOST'] = databricks_host
  os.environ['DATABRICKS_TOKEN'] = databricks_token
  
  with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, 
                                       cur_shard=hvd.rank(), shard_count=hvd.size(),
                                       batch_size=BATCH_SIZE) as train_reader, \
       converter_val.make_tf_dataset(transform_spec=transform_spec_fn, 
                                     cur_shard=hvd.rank(), shard_count=hvd.size(),
                                     batch_size=BATCH_SIZE) as test_reader:
     # tf.keras only accept tuples, not namedtuples
      train_dataset = train_reader.map(lambda x: (x.features, x.label))
      steps_per_epoch = len(converter_train) // (BATCH_SIZE * hvd.size())

      test_dataset = test_reader.map(lambda x: (x.features, x.label))
      
      validation_steps = max(1, len(converter_val) // (BATCH_SIZE * hvd.size()))
      model = build_model(dropout=0.5)

      optimizer = Nadam(lr=0.016)
      optimizer = hvd.DistributedOptimizer(optimizer)

      model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

      callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                   hvd.callbacks.MetricAverageCallback(),
                   EarlyStopping(patience=3, monitor='val_acc', min_delta=0.001, restore_best_weights=True, verbose=(1 if hvd.rank() == 0 else 0))]
      # Record Start time
      start_time = time.time()

      if hvd.rank() == 0:
        callbacks.append(ModelCheckpoint(checkpoint_path + "/checkpoint-{epoch}.ckpt", save_weights_only=True, verbose=1))

      # See comment above on batch_size * num_gpus
      model.fit(train_dataset, epochs=hvd_epochs, steps_per_epoch=steps_per_epoch,
                validation_data=test_dataset, validation_steps=validation_steps,
                verbose=(2 if hvd.rank() == 0 else 0), callbacks=callbacks)
      
      # Evaluate our model
      #score = model.evaluate(test_dataset,verbose=0)

      # Record Complete Time and determine Elapsed Time
      complete_time = time.time()
      elapsed_time = complete_time - start_time
      
      if hvd.rank() == 0:        
        # Log events to MLflow
        with mlflow.start_run(run_id = active_run_uuid):
          mlflow.log_params({"mode": process_mode, "epochs": epochs, "batch_size": BATCH_SIZE})

          #mlflow.log_metrics({"Test Loss": score[0], "Test Accuracy": score[1], "Duration": elapsed_time})

          mlflow.keras.log_model(model, "models")

          # Log TF events to MLflow
          mlflow.log_artifacts(output_dir, artifact_path="tf.events")

# COMMAND ----------

# DBTITLE 1,Use Horovod for distributed DL
with mlflow.start_run() as run:
  active_run_uuid = mlflow.active_run().info.run_uuid
  process_mode = "hvd (distributed)"  
  hr = HorovodRunner(np=2)
  hr.run(train_hvd)          

# COMMAND ----------

# MAGIC %md
# MAGIC ## The MlFlow Model Registry
# MAGIC 
# MAGIC The Model registry allows you to Register and Track model versions and manage the model lifecycle (all the way from QA to A/B Testing and to Production)
# MAGIC 
# MAGIC 
# MAGIC ##### Now we want to update the Model Registry by registering our new model. 
# MAGIC 
# MAGIC We can add the best model to the Model Registry and start tracking the model through the deployment process. 

# COMMAND ----------

# make sure your model name is in the widget for this notebook
model_name = "deep_learning_demo"
client = mlflow.tracking.MlflowClient()

client.create_registered_model(model_name)

registered_model = client.get_registered_model(model_name)
registered_model

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can add a model version to this model by creating a version of the best model we trained.

# COMMAND ----------

import time
version_info = client.create_model_version(
  model_name, 
  f"{run.info.artifact_uri}/models", 
  run.info.run_id)

# COMMAND ----------

# MAGIC   %md
# MAGIC Now that the model has been registered, we can move forward with our process of validation, governance, testing, etc. We could trigger the next step in this process by moving the model to the _Staging_ stage in the registry. Using MLflow webhooks, we could trigger an automatic process, notification, or next manual step. 

# COMMAND ----------

staging_model = client.transition_model_version_stage(model_name, version_info.version, stage="Staging")

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's go ahead and transition this _Staging_ model to _Production_.

# COMMAND ----------

production_model = client.transition_model_version_stage(model_name, version_info.version, stage="Production")

# COMMAND ----------


