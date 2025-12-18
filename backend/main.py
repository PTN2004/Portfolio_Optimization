from fastapi import FastAPI
from pydantic import BaseModel
from services import run_ppo
app = FastAPI("Optimization Portfolio Investment using PPO")

