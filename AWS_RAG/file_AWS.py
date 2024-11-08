import boto3
import os
from dotenv import load_dotenv
load_dotenv()

aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
region_name = os.environ.get("AWS_REGION_NAME")


def upload_file_to_s3(file_name, bucket_name, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Create an S3 client
    s3_client = boto3.client('s3',
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key,
                             region_name=region_name)
    
    # Upload file
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)
        print("File uploaded successfully")
    except Exception as e:
        print(f"Error uploading file: {str(e)}")


def check_aws_login():
    try:
        # Create a session with explicit credentials
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        print("Successfully logged in as:", identity['Arn'])
    except Exception as e:
        print("Failed to log in:", e)

def download_file_from_s3(bucket, object_name, file_name):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    s3 = session.resource('s3')
    try:
        # Download the file
        s3.Bucket(bucket).download_file(object_name.strip(), file_name)
        print("File downloaded successfully.")
        return file_name
    except Exception as e:
        print("Error downloading file:", e)

if __name__ == "__main__":
    check_aws_login()
    # upload_file_to_s3('/Users/yonganyu/Desktop/vulnerability-Prediction-GEOG-research-/blog/snow_English_historical_ML_corpus.txt', 'geogresearch')
    file_path = download_file_from_s3('geogresearch', 'thunder_English_modern_ML_corpus.csv', 'thunder_English_modern_ML_corpus.csv')
    print("File downloaded successfully.")
    print (file_path)
