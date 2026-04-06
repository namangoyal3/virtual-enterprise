import os
from datetime import datetime
from typing import List, Optional, Type
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
)

# 1. Groq LLM Configuration
# Llama 3 70B provides the high-reasoning backbone for the swarm.
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.3, # Lower temperature for business precision
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------------------------------------
# PHASE 2: Custom Google Analytics 4 Tool
# -------------------------------------------------------------

class GA4QuerySchema(BaseModel):
    """Input schema for GA4 queries."""
    property_id: str = Field(..., description="The GA4 Property ID to query.")
    metric_name: str = Field("activeUsers", description="The metric to pull (activeUsers, sessions, etc).")
    dimension_name: str = Field("pagePath", description="The dimension to group by (pagePath, city, etc).")
    days_back: int = Field(30, description="How many days of history to retrieve.")

class GoogleAnalytics4Tool(BaseTool):
    name: str = "google_analytics_4_analyzer"
    description: str = (
        "Useful for pulling live traffic and conversion data from Google Analytics 4. "
        "Requires a 'property_id'. Use this to identify underperforming pages or traffic spikes."
    )
    args_schema: Type[BaseModel] = GA4QuerySchema

    def _run(self, property_id: str, metric_name: str = "activeUsers", dimension_name: str = "pagePath", days_back: int = 30) -> str:
        # Check for service account credentials
        creds_path = "service_account.json"
        if not os.path.exists(creds_path):
            return "ERROR: 'service_account.json' not found in workspace. Please provide credentials to use this tool."

        try:
            client = BetaAnalyticsDataClient.from_service_account_info(creds_path)
            
            request = RunReportRequest(
                property=f"properties/{property_id}",
                dimensions=[Dimension(name=dimension_name)],
                metrics=[Metric(name=metric_name)],
                date_ranges=[DateRange(start_date=f"{days_back}daysAgo", end_date="today")],
            )
            
            response = client.run_report(request)
            
            output = f"GA4 Report for Property {property_id} (Last {days_back} days):\n"
            for row in response.rows:
                output += f"- {row.dimension_values[0].value}: {row.metric_values[0].value}\n"
            
            return output if response.rows else "No data found for this period."
            
        except Exception as e:
            return f"GA4 API Error: {str(e)}"

# Instantiate tools
search_tool = DuckDuckGoSearchRun()
ga4_analyzer = GoogleAnalytics4Tool()

# -------------------------------------------------------------
# PHASE 3: Define The Virtual Enterprise Roster (7 Agents)
# -------------------------------------------------------------

# 1. The CEO (The Visionary)
ceo = Agent(
    role="Chief Executive Officer",
    goal="Orchestrate the company departments to solve high-level directives. Use data insights to drive strategy.",
    backstory="You are a data-driven CEO. You don't just guess—you ask the Analyst for data and the Marketing team for market trends before making decisions.",
    tools=[search_tool],
    allow_delegation=True,
    llm=llm,
    verbose=True
)

# 2. The Data Analyst (The Eyes)
analyst = Agent(
    role="Senior Data Analyst",
    goal="Identify traffic drop-offs, low-converting pages, and user behavior patterns using Google Analytics.",
    backstory="You are obsessed with GA4. You find the 'leaks' in the funnel. When traffic dips on a specific article, you sound the alarm for the PM.",
    tools=[ga4_analyzer],
    llm=llm,
    verbose=True
)

# 3. The Product Manager (The Architect)
pm = Agent(
    role="Lead Product Manager",
    goal="Convert analytical insights and business strategy into technical PRDs.",
    backstory="You take raw data from the Analyst and a 'Go' from the CEO to write the perfect specs for the Engineer.",
    llm=llm,
    verbose=True
)

# 4. The Engineer (The Builder)
engineer = Agent(
    role="Lead Software Engineer",
    goal="Build and deploy full-stack features using Next.js and Python. Focus on speed and stability.",
    backstory="You are the muscle of the operation. You implement what's in the PM's PRD without errors.",
    llm=llm,
    verbose=True
)

# 5. Marketing & SEO Manager (The Growth Lead)
marketing = Agent(
    role="Head of Growth & SEO",
    goal="Maximize search presence and brand visibility. Use search tools to find viral trends.",
    backstory="You bridge the gap between technical SEO and viral social marketing. You launch the features the Engineer builds.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

# 6. Sales Representative (The Closer)
sales = Agent(
    role="Direct Sales Lead",
    goal="Identify high-value leads and close potential enterprise or pro users through contextual outreach.",
    backstory="You monitor the web for people with PM-interview pain and offer them the PM Streak solution personally.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

# 7. Customer Success Agent (The Guardian)
cs = Agent(
    role="Head of Customer Success",
    goal="Maintain high user retention. Identify common support patterns and suggest product improvements.",
    backstory="You represent the voice of the existing user. If users are confused, you advocate for a UI change.",
    llm=llm,
    verbose=True
)

# -------------------------------------------------------------
# PHASE 4: The Enterprise Lifecycle (Standard Operating Procedure)
# -------------------------------------------------------------

def execute_company_mission(directive: str, ga4_property_id: Optional[str] = None):
    # 1. CEO Strategic Plan
    task_strategy = Task(
        description=f"CEO Strategy: Evaluate the Board's directive: '{directive}'. Coordinate with the Data Analyst to see if we have relevant traffic data.",
        expected_output="An executive brief outlining the mission and the specific KPIs we are targeting.",
        agent=ceo
    )

    # 2. Data Analysis (Optional GA4 check)
    analysis_desc = f"Analyst: Pull the last 30 days of top pages for Property {ga4_property_id} and identify where we should focus our efforts to fulfill the CEO's KPI." if ga4_property_id else "Analyst: Research public competition trends and identify where we have the most leverage."
    task_analysis = Task(
        description=analysis_desc,
        expected_output="A list of 3 high-priority focus areas backed by data (either internal GA4 or external research).",
        agent=analyst,
        context=[task_strategy]
    )

    # 3. Product Requirement (PRD)
    task_prd = Task(
        description="PM: Write a full Markdown PRD for a new feature or optimization based on the Analyst's high-priority focus areas.",
        expected_output="A structured PRD with User Stories, Tech Specs, and QA criteria.",
        agent=pm,
        context=[task_analysis]
    )

    # 4. Implementation (Code)
    task_coding = Task(
        description="Engineer: Implement the full Next.js/Python code for the feature defined in the PM's PRD.",
        expected_output="Multiple file blocks containing functional, production-ready code.",
        agent=engineer,
        context=[task_prd]
    )

    # 5. Marketing Launch
    task_marketing = Task(
        description="Marketing: Write a multi-channel launch plan (Twitter, LinkedIn) and generate the SEO keywords we need to target to rank for this new update.",
        expected_output="Social media copy and an SEO keyword target list.",
        agent=marketing,
        context=[task_coding]
    )

    # Create the Virtual Enterprise (Swarm)
    pm_streak_corp = Crew(
        agents=[ceo, analyst, pm, engineer, marketing, sales, cs],
        tasks=[task_strategy, task_analysis, task_prd, task_coding, task_marketing], # Unified workflow
        process=Process.sequential,
        verbose=True
    )

    return pm_streak_corp.kickoff()

if __name__ == "__main__":
    # Test directive with the real PM Streak GA4 Property ID
    ga4_id = "529697573"
    directive = "Analyze why user conversion for the 'Pro' upgrade is lower in the 'Learn' vertical compared to the 'Dashboard' and propose a data-driven UX improvement."
    
    print(f"\n🚀 STARTING ENTERPRISE MISSION: {directive}")
    print(f"📊 TARGETING GA4 PROPERTY: {ga4_id}")
    
    report = execute_company_mission(directive, ga4_id)
    
    print("\n\n========================================")
    print("FINAL STRATEGIC DELIVERABLE:")
    print("========================================")
    print(report)
