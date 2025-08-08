#Pydantic models... 

from pydantic import BaseModel
from typing import List, Union, Dict, Any

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[Union[str, Dict[str, Any]]]