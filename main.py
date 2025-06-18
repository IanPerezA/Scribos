from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware   # ⬅️ 1. importa el middleware
from router.sugerencias_router import router as sugerencias_router

app = FastAPI(title="Scribos API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_methods=["*"],     
    allow_headers=["*"],     
    allow_credentials=True, 
)

app.include_router(sugerencias_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
