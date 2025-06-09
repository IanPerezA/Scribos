from fastapi import FastAPI
from router.sugerencias_router import router as sugerencias_router

app = FastAPI(title="Scribos API")
app.include_router(sugerencias_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)

