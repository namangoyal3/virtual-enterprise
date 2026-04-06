import os
import json
from typing import Optional, Type
from pydantic import BaseModel, Field

# ── GA4 Tool ────────────────────────────────────────────────────

class GA4QuerySchema(BaseModel):
    property_id: str = Field(..., description="The GA4 Property ID.")
    metric_name: str = Field("activeUsers", description="Metric to pull.")
    dimension_name: str = Field("pagePath", description="Dimension to group by.")
    days_back: int = Field(30, description="Days of history.")


def make_ga4_tool():
    from crewai.tools import BaseTool

    class GoogleAnalytics4Tool(BaseTool):
        name: str = "google_analytics_4_analyzer"
        description: str = (
            "Pulls live traffic data from Google Analytics 4. "
            "Use to find underperforming pages or traffic spikes."
        )
        args_schema: Type[BaseModel] = GA4QuerySchema

        def _run(self, property_id: str, metric_name: str = "activeUsers",
                 dimension_name: str = "pagePath", days_back: int = 30) -> str:
            from google.analytics.data_v1beta import BetaAnalyticsDataClient
            from google.analytics.data_v1beta.types import (
                DateRange, Dimension, Metric, RunReportRequest,
            )
            creds_path = os.path.join(os.path.dirname(__file__), "service_account.json")
            if not os.path.exists(creds_path):
                return "ERROR: service_account.json not found."
            try:
                with open(creds_path) as f:
                    info = json.load(f)
                client = BetaAnalyticsDataClient.from_service_account_info(info)
                request = RunReportRequest(
                    property=f"properties/{property_id}",
                    dimensions=[Dimension(name=dimension_name)],
                    metrics=[Metric(name=metric_name)],
                    date_ranges=[DateRange(start_date=f"{days_back}daysAgo", end_date="today")],
                )
                response = client.run_report(request)
                output = f"GA4 Report - Property {property_id} (Last {days_back}d):\n"
                for row in response.rows:
                    output += f"  {row.dimension_values[0].value}: {row.metric_values[0].value}\n"
                return output if response.rows else "No data found."
            except Exception as e:
                return f"GA4 API Error: {e}"

    return GoogleAnalytics4Tool()


# ── Mission Execution (lazy init) ──────────────────────────────

def execute_company_mission(directive: str, ga4_property_id: Optional[str] = None):
    from crewai import Agent, Task, Crew, Process, LLM

    # OpenRouter has much higher rate limits than Groq free tier (12K TPM).
    # Use OpenRouter as the provider, Groq as fallback.
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if openrouter_key:
        llm = LLM(
            model="openrouter/meta-llama/llama-3.3-70b-instruct",
            temperature=0.3,
            api_key=openrouter_key,
            max_tokens=1024,
        )
    else:
        llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=1024,
        )

    ga4_tool = make_ga4_tool()

    ceo = Agent(
        role="Chief Executive Officer",
        goal="Orchestrate departments to solve high-level business directives using data-driven strategy.",
        backstory="You are a data-driven CEO who coordinates analysts, PMs, and engineers before making decisions.",
        allow_delegation=True, llm=llm, verbose=True,
    )
    analyst = Agent(
        role="Senior Data Analyst",
        goal="Identify traffic drop-offs, low-converting pages, and user behavior using GA4.",
        backstory="You are obsessed with metrics. When traffic dips, you sound the alarm.",
        tools=[ga4_tool], llm=llm, verbose=True,
    )
    pm = Agent(
        role="Lead Product Manager",
        goal="Convert analytical insights into technical PRDs with clear specs.",
        backstory="You take raw data from the Analyst and write specs the Engineer can execute.",
        llm=llm, verbose=True,
    )
    engineer = Agent(
        role="Lead Software Engineer",
        goal="Build production-ready features using Next.js and Python based on the PRD.",
        backstory="You implement what is in the PRD with clean, tested, deployable code.",
        llm=llm, verbose=True,
    )
    marketing = Agent(
        role="Head of Growth and SEO",
        goal="Maximize search presence and brand visibility for new features.",
        backstory="You bridge technical SEO and viral social marketing to launch features.",
        llm=llm, verbose=True,
    )
    sales = Agent(
        role="Direct Sales Lead",
        goal="Identify high-value leads and close potential enterprise or pro users.",
        backstory="You monitor the web for PM-interview pain and offer solutions.",
        llm=llm, verbose=True,
    )
    cs = Agent(
        role="Head of Customer Success",
        goal="Maintain high user retention and advocate for UI improvements.",
        backstory="You represent the voice of the user. If users are confused, you advocate for change.",
        llm=llm, verbose=True,
    )

    task_strategy = Task(
        description=f"CEO: Evaluate the directive: '{directive}'. Outline KPIs and coordinate the team.",
        expected_output="Executive brief with mission goals and target KPIs.",
        agent=ceo,
    )
    analysis_desc = (
        f"Pull last 30 days of top pages for GA4 Property {ga4_property_id} and identify focus areas."
        if ga4_property_id
        else "Research competition trends and identify highest-leverage areas."
    )
    task_analysis = Task(
        description=f"Analyst: {analysis_desc}",
        expected_output="3 high-priority focus areas backed by data.",
        agent=analyst, context=[task_strategy],
    )
    task_prd = Task(
        description="PM: Write a Markdown PRD for a feature or optimization based on the Analyst findings.",
        expected_output="Structured PRD with User Stories, Tech Specs, and QA criteria.",
        agent=pm, context=[task_analysis],
    )
    task_coding = Task(
        description="Engineer: Implement production-ready code for the feature in the PRD.",
        expected_output="Functional code blocks ready for deployment.",
        agent=engineer, context=[task_prd],
    )
    task_marketing = Task(
        description="Marketing: Write a multi-channel launch plan and SEO keyword targets.",
        expected_output="Social media copy and SEO keyword target list.",
        agent=marketing, context=[task_coding],
    )

    corp = Crew(
        agents=[ceo, analyst, pm, engineer, marketing, sales, cs],
        tasks=[task_strategy, task_analysis, task_prd, task_coding, task_marketing],
        process=Process.sequential, verbose=True,
    )
    result = corp.kickoff()
    return str(result)
