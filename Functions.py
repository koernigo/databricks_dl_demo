# Databricks notebook source
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


