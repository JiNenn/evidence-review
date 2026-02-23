import boto3
from botocore.exceptions import ClientError
from botocore.client import Config

from app.config import get_settings

settings = get_settings()


def get_s3_client(endpoint_url: str | None = None):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def presign_put(object_key: str, content_type: str) -> str:
    client = get_s3_client(settings.s3_public_endpoint)
    return client.generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.s3_bucket, "Key": object_key, "ContentType": content_type},
        ExpiresIn=settings.s3_presign_expires,
    )


def presign_get(object_key: str) -> str:
    client = get_s3_client(settings.s3_public_endpoint)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": object_key},
        ExpiresIn=settings.s3_presign_expires,
    )


def ensure_bucket_exists() -> None:
    client = get_s3_client()
    buckets = client.list_buckets().get("Buckets", [])
    if any(bucket["Name"] == settings.s3_bucket for bucket in buckets):
        return
    client.create_bucket(Bucket=settings.s3_bucket)


def get_object_bytes(object_key: str) -> bytes:
    client = get_s3_client()
    response = client.get_object(Bucket=settings.s3_bucket, Key=object_key)
    return response["Body"].read()


def object_exists(object_key: str) -> bool:
    client = get_s3_client()
    try:
        client.head_object(Bucket=settings.s3_bucket, Key=object_key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def put_object_bytes(object_key: str, payload: bytes, content_type: str = "application/octet-stream") -> None:
    client = get_s3_client()
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=object_key,
        Body=payload,
        ContentType=content_type,
    )
