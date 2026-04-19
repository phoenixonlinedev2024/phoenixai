"""Object storage tools — S3, R2 (Cloudflare free), MinIO via boto3."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry

from jarvis.config import cfg


def _get_s3():
    try:
        import boto3
        kwargs = dict(
            aws_access_key_id=cfg.S3_ACCESS_KEY,
            aws_secret_access_key=cfg.S3_SECRET_KEY,
            region_name=cfg.S3_REGION,
        )
        if cfg.S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = cfg.S3_ENDPOINT_URL
        return boto3.client("s3", **kwargs)
    except ImportError:
        raise RuntimeError("boto3 not installed. Run: pip install boto3")


def _s3_list(bucket: str, prefix: str = "") -> str:
    try:
        s3 = _get_s3()
        kwargs = {"Bucket": bucket}
        if prefix:
            kwargs["Prefix"] = prefix
        resp = s3.list_objects_v2(**kwargs)
        objects = resp.get("Contents", [])
        lines = [f"{o['Key']} ({o['Size']} bytes)" for o in objects[:50]]
        return "\n".join(lines) or "Empty bucket/prefix."
    except Exception as exc:
        return f"S3 list error: {exc}"


def _s3_upload(local_path: str, bucket: str, key: str = "") -> str:
    try:
        s3 = _get_s3()
        k = key or Path(local_path).name
        s3.upload_file(local_path, bucket, k)
        return f"Uploaded {local_path} → s3://{bucket}/{k}"
    except Exception as exc:
        return f"S3 upload error: {exc}"


def _s3_download(bucket: str, key: str, local_path: str = "") -> str:
    try:
        s3 = _get_s3()
        dest = local_path or f"/tmp/{Path(key).name}"
        s3.download_file(bucket, key, dest)
        return f"Downloaded s3://{bucket}/{key} → {dest}"
    except Exception as exc:
        return f"S3 download error: {exc}"


def _s3_delete(bucket: str, key: str) -> str:
    try:
        s3 = _get_s3()
        s3.delete_object(Bucket=bucket, Key=key)
        return f"Deleted s3://{bucket}/{key}"
    except Exception as exc:
        return f"S3 delete error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(name="s3_list", description="List objects in an S3/R2/MinIO bucket.",
        input_schema={"type":"object","properties":{"bucket":{"type":"string"},"prefix":{"type":"string","default":""}},"required":["bucket"]},
        fn=_s3_list, category="files"))
    registry.register(Tool(name="s3_upload", description="Upload a local file to S3/R2/MinIO.",
        input_schema={"type":"object","properties":{"local_path":{"type":"string"},"bucket":{"type":"string"},"key":{"type":"string"}},"required":["local_path","bucket"]},
        fn=_s3_upload, category="files"))
    registry.register(Tool(name="s3_download", description="Download a file from S3/R2/MinIO.",
        input_schema={"type":"object","properties":{"bucket":{"type":"string"},"key":{"type":"string"},"local_path":{"type":"string"}},"required":["bucket","key"]},
        fn=_s3_download, category="files"))
    registry.register(Tool(name="s3_delete", description="Delete an object from S3/R2/MinIO.",
        input_schema={"type":"object","properties":{"bucket":{"type":"string"},"key":{"type":"string"}},"required":["bucket","key"]},
        fn=_s3_delete, category="files"))
