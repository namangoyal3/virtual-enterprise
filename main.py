import os
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from crew import execute_company_mission

app = FastAPI(title="PM Streak Virtual Enterprise API")

# Input Schema
class MissionRequest(BaseModel):
    directive: str
    ga4_property_id: Optional[str] = None
    callback_url: Optional[str] = None # For future webhook integration

@app.get("/")
async def health_check():
    return {"status": "online", "departments": "CEO, Analytics, PM, Engineering, Marketing, Sales, CS"}

@app.post("/api/mission")
async def trigger_mission(request: MissionRequest, background_tasks: BackgroundTasks):
    """
    Spawns the Virtual Enterprise Swarm to tackle a business directive.
    Because multi-agent reasoning takes > 1 minute, we execute in a Background Task.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured in the environment.")

    # We return an immediate 202 status to the client
    # In a full-scale deployment, this would write the final report to a database or ping a webhook
    print(f"Spawning enterprise mission for: {request.directive}")
    
    # Using background task for the actual CrewAI execution
    background_tasks.add_task(execute_company_mission, request.directive, request.ga4_property_id)
    
    return {
        "status": "accepted",
        "message": "CEO has received the mission. The Enterprise departments are now coordinating strategy in the background.",
        "estimated_duration": "3-6 minutes"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
