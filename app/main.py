from fastapi import FastAPI
from app.routers import inference
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,  # cookie 설정
    allow_methods=["*"],
    allow_headers=["*"],
)

# inference 라우터 추가
app.include_router(inference.router)