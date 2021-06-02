# Databricks notebook source
# MAGIC %md
# MAGIC  Use `/dbfs/ml` and Petastorm for Efficient Data Access
# MAGIC  
# MAGIC Petastorm is an open source data access library. This library enables single-node or distributed training and evaluation of deep learning models directly from datasets in Apache Parquet format and datasets that are already loaded as Apache Spark DataFrames. Petastorm supports popular Python-based machine learning (ML) frameworks such as Tensorflow, PyTorch, and PySpark. For more information about Petastorm, refer to the Petastorm GitHub page and Petastorm API documentation.

# COMMAND ----------

dbutils.widgets.text("table_path","/ml/images/tables/")
table_path=dbutils.widgets.get("table_path")
dbutils.widgets.text("image_path","/tmp/256_ObjectCategories/")
dbutils.widgets.text("user","oliver.koernig@databricks.com")
user = dbutils.widgets.get("user")
image_path = dbutils.widgets.get("image_path")
table_path=dbutils.widgets.get("table_path")

# COMMAND ----------

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import Dropout
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
import tensorflow as tf
from tensorflow.data.experimental import unbatch
from tensorflow.io import decode_raw
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pyspark.sql.functions import col

import numpy as np
from sklearn.model_selection import train_test_split


IMG_SHAPE = (299, 299, 3)

# Additional options here will be used later:
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

spark.sql("USE dl_demo")
full_data = spark.table("labeled_images")
train_data, val_data = full_data.randomSplit([0.9, 0.1], seed=12345)

# COMMAND ----------

# Set a cache directory on DBFS FUSE for intermediate data.
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")

converter_train = make_spark_converter(train_data) #add repartition
convertor_val = make_spark_converter(val_data)

# COMMAND ----------

IMG_SHAPE = (299, 299, 3)
img_size = 299

def preprocess(content):
  """
  Preprocess an image file bytes for MobileNetV2 (ImageNet).
  """
  image = Image.open(io.BytesIO(content)).resize([img_size, img_size])
  image_array = keras.preprocessing.image.img_to_array(image)
  return preprocess_input(image_array)

def transform_row(pd_batch):
  """
  The input and output of this function are pandas dataframes.
  """
  pd_batch['features'] = pd_batch['image'].map(lambda x: preprocess(x))
  pd_batch = pd_batch.drop(labels='image', axis=1)
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

# MAGIC %md
# MAGIC From the Parquet table representation, the image data needs some further transformations that are specific to the Keras model used below. They're reshaped again to 299x299(x3), and normalized to [-1,1]. To start, we'll just work with a sample of the train set in memory, and construct a train/test split of the images and labels from there.

# COMMAND ----------

from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
import tensorflow as tf
from tensorflow.data.experimental import unbatch
from tensorflow.io import decode_raw
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

img_size = 299

# Defines a transformation on petastorm data using TF APIs that reshapes and encodes the input.
# This include shuffling and rebatching the data to the desired batch size.
def transform_reader(reader, batch_size):
  def transform_input(x):
    img_bytes = tf.reshape(decode_raw(x.image, tf.uint8), (-1,img_size,img_size,3))
    inputs = preprocess_input(tf.cast(img_bytes, tf.float32))
    outputs = x.label - 1
    return (inputs, outputs)
  # unbatch() is important as the batches from petastorm may vary in size, and have to be 'rebatched'
  # Shuffling across the batches from petastorm mixes up the input a little better
  return make_petastorm_dataset(reader).map(transform_input).apply(unbatch()).shuffle(400, seed=42).batch(batch_size, drop_remainder=True)

# COMMAND ----------

path_base = "/dbfs/ml/images/tables/pq/"
checkpoint_path = path_base + "checkpoint"

table_path_base = path_base
table_path_base_file = "file:" + table_path_base

# cur_shard and shard_count will be used later
def make_caching_reader(suffix, cur_shard=None, shard_count=None):
  return make_batch_reader(table_path_base_file + suffix, num_epochs=None,
                           cur_shard=cur_shard, shard_count=shard_count,
                           cache_type='local-disk', cache_location="/tmp/" + suffix, cache_size_limit=20000000000, # 20GB
                           cache_row_size_estimate=img_size * img_size * 3)

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

# MAGIC %md
# MAGIC We will set the experiment so that we can track our training using MLfFlow. If we did not do this, MLflow would track to a notebook scoped experiment automatically.  

# COMMAND ----------

experiment_name = f"/Users/{user}/Deep Learning Image Demo"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

import os
import tempfile
from sparkdl import HorovodRunner
import random

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

import horovod.tensorflow.keras as hvd
from tensorflow.keras import backend as K
import tensorflow as tf
from sparkdl import HorovodRunner
import math
import time
import json

BATCH_SIZE = 32
num_gpus = 4
epochs = 12
databricks_host = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

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
       convertor_val.make_tf_dataset(transform_spec=transform_spec_fn, 
                                     cur_shard=hvd.rank(), shard_count=hvd.size(),
                                     batch_size=BATCH_SIZE) as test_reader:
     # tf.keras only accept tuples, not namedtuples
      train_dataset = train_reader.map(lambda x: (x.features, x.label))
      steps_per_epoch = len(converter_train) // (BATCH_SIZE * hvd.size())

      test_dataset = test_reader.map(lambda x: (x.features, x.label))
      
      validation_steps = max(1, len(convertor_val) // (BATCH_SIZE * hvd.size()))
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
          mlflow.log_params({"mode": process_mode, "epochs": epochs, "batch_size": batch_size})

          #mlflow.log_metrics({"Test Loss": score[0], "Test Accuracy": score[1], "Duration": elapsed_time})

          mlflow.keras.log_model(model, "models")

          # Log TF events to MLflow
          mlflow.log_artifacts(output_dir, artifact_path="tf.events")

# COMMAND ----------

# DBTITLE 1,Use Horovod for distributed DL
with mlflow.start_run() as run:
  active_run_uuid = mlflow.active_run().info.run_uuid
  process_mode = "hvd (distributed)"  
  hr = HorovodRunner(np=3)
  hr.run(train_hvd)          

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /tmp/256_ObjectCategories/

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

# Wait until the model is ready
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(30):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)
wait_until_ready(version_info.name, version_info.version)

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
