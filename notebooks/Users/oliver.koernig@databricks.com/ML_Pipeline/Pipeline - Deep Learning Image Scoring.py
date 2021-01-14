# Databricks notebook source
# MAGIC %md Main Pipeline for the Ml Pipline Demo Image Scoring jobs

# COMMAND ----------

#Helper Function
def run_with_retry(notebook, timeout, args = {}, max_retries = 3):
  num_retries = 0
  while num_retries <= max_retries:
    try:
      return dbutils.notebook.run(notebook, timeout)
    except Exception as e:
      if num_retries > max_retries:
        raise e
      else:
        print ("Retrying error"), e
        num_retries += 1

# COMMAND ----------

# DBTITLE 1,Step 1: Load New Images For Scoring
dbutils.notebook.run( '/Users/oliver.koernig@databricks.com/ML_Pipeline/Deep Learning Image Prep Scoring',0)

# COMMAND ----------

# DBTITLE 1,Step 2:  Run Scoring Job
dbutils.notebook.run('/Users/oliver.koernig@databricks.com/ML_Pipeline/Deep Learning Image Batch Scoring',0)

# COMMAND ----------

# MAGIC %md Cluster Spec
# MAGIC 
# MAGIC {
# MAGIC     "num_workers": 3,
# MAGIC     "cluster_name": "ok-dl-test-gpu-dbr7",
# MAGIC     "spark_version": "7.3.x-gpu-ml-scala2.12",
# MAGIC     "spark_conf": {},
# MAGIC     "aws_attributes": {
# MAGIC         "first_on_demand": 1,
# MAGIC         "availability": "SPOT_WITH_FALLBACK",
# MAGIC         "zone_id": "us-west-2b",
# MAGIC         "spot_bid_price_percent": 100,
# MAGIC         "ebs_volume_count": 0
# MAGIC     },
# MAGIC     "node_type_id": "g4dn.12xlarge",
# MAGIC     "driver_node_type_id": "g4dn.12xlarge",
# MAGIC     "ssh_public_keys": [],
# MAGIC     "custom_tags": {},
# MAGIC     "spark_env_vars": {},
# MAGIC     "autotermination_minutes": 20,
# MAGIC     "enable_elastic_disk": false,
# MAGIC     "cluster_source": "UI",
# MAGIC     "init_scripts": [],
# MAGIC     "cluster_id": "1116-122141-mien87"
# MAGIC }