from fastapi import APIRouter
from pydantic import BaseModel
from controller.sugerencias_controller import get_suggestions

router = APIRouter(prefix="/suggest", tags=["suggest"])

class SuggestReq(BaseModel):
    pattern: list[str]

@router.post("", response_model=list[str])
def suggest(req: SuggestReq):
    return get_suggestions(req.pattern)
