# Databricks notebook source
# MAGIC %sql
# MAGIC drop table if exists dl_demo.image_data

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE dl_demo.image_data 
# MAGIC (path STRING
# MAGIC , modificationTime TIMESTAMP
# MAGIC , length LONG
# MAGIC , content BINARY
# MAGIC , label INT
# MAGIC , load_date DATE
# MAGIC , predicted_label INT)
# MAGIC USING DELTA
# MAGIC LOCATION '/tmp/dl_demo/images_data'

# COMMAND ----------

# MAGIC %sql
# MAGIC describe extended  dl_demo.image_data

# COMMAND ----------

# later, add in all of the copy image piece, and then move all image 001 into a new image fodler for later
