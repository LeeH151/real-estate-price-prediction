from pydantic import BaseModel
from typing import Literal

class User(BaseModel):
    id: str
    name: str
    email: str
    password: str
    role: Literal["user", "admin"]
    status: Literal["active", "locked"]