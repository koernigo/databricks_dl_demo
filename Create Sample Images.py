# Databricks notebook source
# DBTITLE 1,Install Kaggle
# MAGIC %pip install kaggle

# COMMAND ----------

# DBTITLE 1,Install Folder for Kaggle authentication
# MAGIC %sh
# MAGIC mkdir /root/.kaggle

# COMMAND ----------

# DBTITLE 1,Create Kaggle Json. Use your own username and kaggle key
# MAGIC %sh
# MAGIC echo "{\"username\":\"<KAGGLE_USERNAME>\",\"key\":\"<KAGGLE_KEY>\"}" > /root/.kaggle/kaggle.json

# COMMAND ----------

# DBTITLE 1,Check that the file is there and has the correct format
# MAGIC %sh
# MAGIC ls /root/.kaggle
# MAGIC cat /root/.kaggle/kaggle.json

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir /dbfs/tmp

# COMMAND ----------

# MAGIC %sh
# MAGIC chmod 600 /root/.kaggle/kaggle.json

# COMMAND ----------

# DBTITLE 1,Download Kaggle Dataset to dbfs root
# MAGIC %sh
# MAGIC cd /dbfs
# MAGIC kaggle datasets download -d jessicali9530/caltech256 --force

# COMMAND ----------

# DBTITLE 1,Verify that File has been successfully downloaded
# MAGIC %sh
# MAGIC ls /dbfs/caltech256.zip

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /dbfs/tmp
# MAGIC unzip ../caltech256.zip 

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/tmp

# COMMAND ----------

# DBTITLE 1,Uncomment for Cleanup
#%sh
#rm -rf /dbfs/tmp
#rm /dbfs/caltech256.zip
