from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Schema 基类"""
    model_config = ConfigDict(
        populate_by_name=True,
    )
