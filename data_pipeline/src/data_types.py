from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AssistantResponse(BaseModel):
    tool_logs: list = Field(default_factory=list)
    messages: list = Field(default_factory=list)


class UserMessage(BaseModel):
    content: str = ""
    responses: list[AssistantResponse] = Field(default_factory=list)
    is_conflicting: bool = False


class Example(BaseModel):
    id: str = ""
    title: str = ""
    description: str = ""
    instructions: str = ""
    source: str = ""
    clauses: list[str] = Field(default_factory=list)
    is_test: bool = False
    user_messages: list[UserMessage] = Field(default_factory=list)
    error: Optional[str] = None

    model_config = ConfigDict(extra="allow")