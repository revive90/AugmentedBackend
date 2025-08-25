from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
def ping():
    return {"ok": True, "msg": "pong"}


#C:\Users\924139\PycharmProjects\NewPrototypeAug\data\demodata
#C:\Users\924139\PycharmProjects\NewPrototypeAug\data\demodataout