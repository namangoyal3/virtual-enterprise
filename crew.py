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

def make_github_pr_tool():
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field
    from typing import Type
    
    class GitHubPRSchema(BaseModel):
        repo_name: str = Field(..., description="Repository name, e.g. 'namangoyal3/pm-streak'")
        file_path: str = Field(..., description="Path to the file to update, e.g. 'src/app/page.tsx'")
        new_content: str = Field(..., description="Complete newly updated code for the file")
        commit_message: str = Field(..., description="Commit message describing the change")
        pr_title: str = Field(..., description="Title for the Pull Request")
        pr_body: str = Field(..., description="Description/PR body explaining the feature")

    class GitHubPRTool(BaseTool):
        name: str = "github_pr_creator"
        description: str = (
            "Creates a new branch, commits updated code to a file, and opens a Pull Request on GitHub. "
            "Use this tool when you want to deploy a feature live or push an A/B test."
        )
        args_schema: Type[BaseModel] = GitHubPRSchema

        def _run(self, repo_name: str, file_path: str, new_content: str, commit_message: str, pr_title: str, pr_body: str) -> str:
            import os
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                return "ERROR: GITHUB_TOKEN not found in environment."
            
            try:
                from github import Github
                import time
                g = Github(token)
                repo = g.get_repo(repo_name)
                
                main_ref = repo.get_git_ref("heads/main")
                branch_name = f"ai-feature-{int(time.time())}"
                repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=main_ref.object.sha)
                
                try:
                    contents = repo.get_contents(file_path, ref="main")
                    repo.update_file(contents.path, commit_message, new_content, contents.sha, branch=branch_name)
                except Exception:
                    repo.create_file(file_path, commit_message, new_content, branch=branch_name)
                    
                pr = repo.create_pull(title=pr_title, body=pr_body, head=branch_name, base="main")
                return f"SUCCESS: Pull request created! URL: {pr.html_url}"
            except Exception as e:
                return f"GitHub API Error: {e}"

    return GitHubPRTool()

# ── Mission Execution (lazy init) ──────────────────────────────

def execute_company_mission(directive: str, ga4_property_id: Optional[str] = None):
    from crewai import Agent, Task, Crew, Process, LLM

    # Using Llama 4 Scout (newer model, likely higher rate limits on Groq free tier)
    llm = LLM(
        model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=512,
    )

    ga4_tool = make_ga4_tool()
    github_tool = make_github_pr_tool()

    ceo = Agent(
        role="CEO (Chief Executive Officer)",
        goal="Set company vision, orchestrate the C-suite, and make final strategic calls using data-driven insights.",
        backstory="You are a visionary CEO who has scaled multiple startups to $100M ARR. You never guess — you demand data from the CDO and market intel from the CMO before making any decision.",
        allow_delegation=True, llm=llm, verbose=True,
    )
    cdo = Agent(
        role="CDO (Chief Data Officer)",
        goal="Own the entire data strategy. Use GA4 to identify traffic drop-offs, funnel leaks, and user behavior patterns.",
        backstory="You are a CDO who built the data infrastructure at two unicorns. You live in dashboards and can spot a conversion anomaly before anyone else in the room.",
        tools=[ga4_tool], llm=llm, verbose=True,
    )
    cpo = Agent(
        role="CPO (Chief Product Officer)",
        goal="Own the product roadmap. Convert data insights and CEO strategy into concrete PRDs with user stories and success metrics.",
        backstory="You are a CPO who shipped products used by 10M+ people. You turn ambiguous business goals into razor-sharp specs that engineers love.",
        llm=llm, verbose=True,
    )
    cto = Agent(
        role="CTO (Chief Technology Officer)",
        goal="Architect and ship production-ready code using Next.js and Python. Deploy code using GitHub Pull Requests.",
        backstory="You are a CTO who has built systems handling 1M+ requests per second. You use the github_pr_creator tool to deploy features automatically to 'namangoyal3/pm-streak'.",
        tools=[github_tool], llm=llm, verbose=True,
    )
    cqo = Agent(
        role="CQO (Chief Quality & Testing Officer)",
        goal="Design rigorous A/B tests for all new features and QA the CTO's implementations.",
        backstory="You are obsessed with continuous testing and statistical significance. You evaluate PRs and dictate which successful features should be kept and optimized.",
        llm=llm, verbose=True,
    )
    cmo = Agent(
        role="CMO (Chief Marketing Officer)",
        goal="Own brand, growth, and SEO. Maximize search rankings and create viral launch campaigns across all channels.",
        backstory="You are a CMO who grew a SaaS from 0 to 500K organic monthly visitors. You think in funnels, keywords, and distribution channels.",
        llm=llm, verbose=True,
    )
    cro = Agent(
        role="CRO (Chief Revenue Officer)",
        goal="Own the revenue pipeline. Identify high-value enterprise leads et close deals through strategic outreach.",
        backstory="You are a CRO who has closed $50M+ in enterprise SaaS deals. You find pain points in the market and position the product as the solution.",
        llm=llm, verbose=True,
    )
    cco = Agent(
        role="CCO (Chief Customer Officer)",
        goal="Own retention, NPS, and the voice of the customer. Turn support patterns into product improvement proposals.",
        backstory="You are a CCO who maintained 95%+ retention at a B2B SaaS. You obsess over every confused user and turn their pain into actionable product feedback.",
        llm=llm, verbose=True,
    )

    task_strategy = Task(
        description=f"CEO: Evaluate the board directive: '{directive}'. Set KPIs and assign ownership to the C-suite.",
        expected_output="Executive brief with mission goals, target KPIs, and department assignments.",
        agent=ceo,
    )
    analysis_desc = (
        f"CDO: Pull last 30 days of top pages for GA4 Property {ga4_property_id} and identify focus areas."
        if ga4_property_id
        else "CDO: Research competition trends and identify highest-leverage areas."
    )
    task_analysis = Task(
        description=analysis_desc,
        expected_output="3 high-priority focus areas backed by data.",
        agent=cdo, context=[task_strategy],
    )
    task_prd = Task(
        description="CPO: Write a Markdown PRD for a feature or optimization based on the CDO's findings.",
        expected_output="Structured PRD with User Stories, Tech Specs, and QA criteria.",
        agent=cpo, context=[task_analysis],
    )
    task_coding = Task(
        description="CTO: Implement production-ready code for the CPO's PRD. You MUST open a Pull Request using the github_pr_creator tool to deploy this feature to 'namangoyal3/pm-streak'. Provide the PR URL when done.",
        expected_output="A successful PR URL with the implemented features.",
        agent=cto, context=[task_prd],
    )
    task_qa = Task(
        description="CQO: Review the CTO's implementation/PR. Design a rigorous A/B testing plan. Define success criteria to ensure that only successful features are kept.",
        expected_output="QA validation report and A/B test launch plan.",
        agent=cqo, context=[task_coding],
    )
    task_marketing = Task(
        description="CMO: Write a multi-channel launch plan and SEO keyword targets for the new feature.",
        expected_output="Social media copy and SEO keyword target list.",
        agent=cmo, context=[task_coding],
    )

    corp = Crew(
        agents=[ceo, cdo, cpo, cto, cqo, cmo, cro, cco],
        tasks=[task_strategy, task_analysis, task_prd, task_coding, task_qa, task_marketing],
        process=Process.sequential, verbose=True,
    )
    result = corp.kickoff()
    return str(result)
