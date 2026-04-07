"""
infrastructure/s3_setup.py
Creates and validates the S3 bucket structure for the street-risk-mvp pipeline.
Run once: python infrastructure/s3_setup.py
"""

import boto3
import os
from pathlib import Path
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Anchor to repo root regardless of where the script is invoked from
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Medallion layer prefixes — these are logical folders inside S3
PREFIXES = [
    "bronze/images/",
    "bronze/crash/",
    "bronze/roads/",
    "silver/road_points/",
    "silver/image_features/",
    "silver/crash_hex/",
    "gold/training_table/",
    "gold/scored_hexagons/",
    "mlflow/",
]


def create_bucket(s3_client):
    try:
        if REGION == "us-east-1":
            s3_client.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3_client.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        print(f"✓ Bucket created: s3://{BUCKET_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
            print(f"✓ Bucket already exists and is yours: s3://{BUCKET_NAME}")
        else:
            raise


def create_prefix_placeholders(s3_client):
    """
    S3 has no real folders — we write empty placeholder objects
    to make the structure visible in the console.
    """
    for prefix in PREFIXES:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=prefix)
        print(f"  ✓ Created prefix: {prefix}")


def validate_bucket(s3_client):
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    keys = [obj["Key"] for obj in response.get("Contents", [])]
    print(f"\n✓ Bucket validated — {len(keys)} prefixes found:")
    for k in keys:
        print(f"  s3://{BUCKET_NAME}/{k}")


if __name__ == "__main__":
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    print(f"\nInitializing S3 bucket: {BUCKET_NAME}\n")
    create_bucket(s3)
    create_prefix_placeholders(s3)
    validate_bucket(s3)
    print("\n✓ Infrastructure ready.\n")