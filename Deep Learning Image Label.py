# Databricks notebook source
# MAGIC %md
# MAGIC Demo the callout to the AWS A2I service. Note that this is not doing an actual calout , however it contains all the required steps.

# COMMAND ----------

# MAGIC %sql
# MAGIC use dl_demo

# COMMAND ----------

import boto3

client = boto3.client('sagemaker-a2i-runtime', 'east-us-2')

# COMMAND ----------

# MAGIC %md Create Input content Json objects

# COMMAND ----------

import json
#read new images to be processed from label table
new_images_tobe_labeled = spark.sql("select * from image_data where predicted_score < 95 and predicted_score is not null")

#Write to json
result = new_images_tobe_labeled \
  .select("path","predicted_label") \
  .write.mode("overwrite") \
  .option("multiline","false") \
  .json("/tmp/new_images.json")

json_images = new_images_tobe_labeled.toJSON().collect()

# COMMAND ----------

import json
input_content_str = json.dumps(json_images)

# COMMAND ----------

print(input_content_str)

# COMMAND ----------

 '''
 try:
   
  response = client.start_human_loop(
    HumanLoopName='Manual_Image_Flow',
    FlowDefinitionArn='arn:......',
    HumanLoopInput={
        'InputContent': input_content_str
    },
    DataAttributes={
        'ContentClassifiers': [
            'FreeOfPersonallyIdentifiableInformation','FreeOfAdultContent',
        ]
    }
   )
 
  except client.meta.client.exceptions.ConflictException as err:
    print("Your request has the same name as another active human loop but has different input data. You cannot start two human loops with the same name and different input data']))
    raise err
   '''

# COMMAND ----------

response = {'HumanLoopArn': 'arn:aws:sagemaker:us-east-1:xxxxxxxx:flow-definition/fd-sagemaker-object-detection-demo-2020-05-01-18-08-47'}

# COMMAND ----------

'''completed_human_loops = []
for human_loop_name in human_loops_started:
    resp = a2i.describe_human_loop(HumanLoopName=human_loop_name)
    print(f'HumanLoop Name: {human_loop_name}')
    print(f'HumanLoop Status: {resp["HumanLoopStatus"]}')
    print(f'HumanLoop Output Destination: {resp["HumanLoopOutput"]}')
    print('\n')
    
    if resp["HumanLoopStatus"] == "Completed":
        completed_human_loops.append(resp)'''

# COMMAND ----------

'''import re
import pprint

pp = pprint.PrettyPrinter(indent=4)

for resp in completed_human_loops:
    splitted_string = re.split('s3://' +  BUCKET + '/', resp['HumanLoopOutput']['OutputS3Uri'])
    output_bucket_key = splitted_string[1]

    response = s3.get_object(Bucket=BUCKET, Key=output_bucket_key)
    content = response["Body"].read()
    json_output = json.loads(content)
    pp.pprint(json_output)
    print('\n')'''

# COMMAND ----------

#last step : Insert manually labeled images back into the main Delta Table  image_label_results with a indicator that they are manually labeled. These can then flow into the labeled_images table as well !!!

# COMMAND ----------

# MAGIC %md
# MAGIC response = {'human': 'arn:aws:sagemaker:us-east-1:xxxxxxxx:flow-definition/fd-sagemaker-object-detection-demo-2020-05-01-18-08-47', 'humanAnswers': [{'answerContent': { 'annotatedResult': { 'boundingBoxes': [{'height': 1801, 'label': 'bicycle', 'left': 1042, 'top': 627, 'width': 2869}], 'inputImageProperties': {'height': 2608, 'width': 3911}}}, 'submissionTime': '2020-05-01T18:24:53.742Z', 'workerId': 'xxxxxxxxxx'}], 'humanLoopName': 'xxxxxxx-xxxxx-xxxx-xxxx-xxxxxxxxxx', 'inputContent': {'initialValue': 'bicycle', 'taskObject': 's3://sagemaker-us-east-1-xxxxxxx/a2i-results/sample-a2i-images/pexels-photo-276517.jpeg'}}

# COMMAND ----------

# MAGIC %md
# MAGIC Errors
# MAGIC For information about the errors that are common to all actions, see Common Errors.
# MAGIC 
# MAGIC ConflictException
# MAGIC Your request has the same name as another active human loop but has different input data. You cannot start two human loops with the same name and different input data.
# MAGIC 
# MAGIC HTTP Status Code: 409
# MAGIC 
# MAGIC InternalServerException
# MAGIC We couldn't process your request because of an issue with the server. Try again later.
# MAGIC 
# MAGIC HTTP Status Code: 500
# MAGIC 
# MAGIC ServiceQuotaExceededException
# MAGIC You exceeded your service quota. Service quotas, also referred to as limits, are the maximum number of service resources or operations for your AWS account. For a list of Amazon A2I service quotes, see Amazon Augmented AI Service Quotes. Delete some resources or request an increase in your service quota. You can request a quota increase using Service Quotas or the AWS Support Center. To request an increase, see AWS Service Quotas in the AWS General Reference.
# MAGIC 
# MAGIC HTTP Status Code: 402
# MAGIC 
# MAGIC ThrottlingException
# MAGIC You exceeded the maximum number of requests.
# MAGIC 
# MAGIC HTTP Status Code: 429
# MAGIC 
# MAGIC ValidationException
# MAGIC The request isn't valid. Check the syntax and try again.
# MAGIC 
# MAGIC HTTP Status Code: 400
