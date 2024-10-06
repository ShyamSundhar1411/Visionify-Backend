from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    TF_ENABLE_ONEDNN_OPTS: int = 0
