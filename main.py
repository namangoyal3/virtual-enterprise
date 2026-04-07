import os
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from crew import execute_company_mission

app = FastAPI(title="PM Streak Virtual Enterprise API")

class MissionRequest(BaseModel):
    directive: str
    ga4_property_id: Optional[str] = None

# Store results in-memory (upgrade to DB for prod)
mission_results: dict = {}
mission_counter = 0

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "departments": ["CEO", "CDO (Data)", "CPO (Product)", "CTO (Tech)", "CQO (Quality)", "CMO (Marketing)", "CRO (Revenue)", "CCO (Customer)"],
        "missions_completed": len(mission_results),
    }

def run_mission(mission_id: int, directive: str, ga4_property_id: Optional[str]):
    try:
        result = execute_company_mission(directive, ga4_property_id)
        mission_results[mission_id] = {"status": "completed", "result": result}
        print(f"✅ Mission {mission_id} completed.")
    except Exception as e:
        mission_results[mission_id] = {"status": "failed", "error": str(e)}
        print(f"❌ Mission {mission_id} failed: {e}")

@app.post("/api/mission")
async def trigger_mission(request: MissionRequest, background_tasks: BackgroundTasks):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    global mission_counter
    mission_counter += 1
    mid = mission_counter
    mission_results[mid] = {"status": "running"}

    background_tasks.add_task(run_mission, mid, request.directive, request.ga4_property_id)

    return {
        "status": "accepted",
        "mission_id": mid,
        "message": "CEO has received the mission. Departments are coordinating.",
        "check_result": f"/api/mission/{mid}",
    }

@app.get("/api/mission/{mission_id}")
async def get_mission_result(mission_id: int):
    if mission_id not in mission_results:
        raise HTTPException(status_code=404, detail="Mission not found.")
    return mission_results[mission_id]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
