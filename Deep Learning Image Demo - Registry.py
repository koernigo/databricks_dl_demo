# Databricks notebook source
# MAGIC %md
# MAGIC ##Retrieve Run from Tracking Server

# COMMAND ----------

dbutils.widgets.text("model_name","deep_learning_demo")
model_name = dbutils.widgets.get("model_name")
dbutils.widgets.text("experiment_name","DL_Image_Classification")
experiment_name = dbutils.widgets.get("experiment_name")
dbutils.widgets.text("job_user","oliver.koernig@databricks.com")
job_user=dbutils.widgets.get("job_user")

# COMMAND ----------

import mlflow
print("MLflow Version: %s" % mlflow.__version__)

# COMMAND ----------

#Find Experiment path for the given name and set the experiment
experiment_path=f"/Users/{job_user}/{experiment_name}"
print (experiment_path)
mlflow.set_experiment(experiment_path)

# COMMAND ----------

mlflow.search_runs().loc(0)

# COMMAND ----------

#get the run_id from the latest run within the given experiment and retrieve that run
run_id = mlflow.search_runs().iloc[0,0]
print(run_id)
run = mlflow.get_run(run_id)

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
client = mlflow.tracking.MlflowClient()
try:
  client.create_registered_model(model_name)
except:
  print("Model already exists, ignoring create model statement")
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
