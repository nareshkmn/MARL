# research_agents.py
import json
from datetime import datetime
from typing import List, Dict, Any

from crewai import Agent, Task, Crew, Process
import networkx as nx
import plotly.graph_objects as go

from config import Config
from gemini_scholar import build_corpus
import os
os.environ["CREWAI_USE_NATIVE_GEMINI"] = "false"


class ResearchAgentSystem:
    """
    Multi-agent research lab:
    - Uses Gemini (2.5-flash) via LiteLLM-compatible model string
    - Ingests a corpus of papers via gemini_scholar.build_corpus
    - Provides that corpus to agents as shared context
    """

    def __init__(self, topic: str):
        self.topic = topic

        # CrewAI expects a model string; LiteLLM will handle Gemini
        self.llm_model_name = Config.GEMINI_LITELLM_MODEL

        self.reasoning_graph = nx.DiGraph()
        self.conversation_history: List[Dict[str, Any]] = []
        self.agent_interactions: List[Dict[str, Any]] = []
        self.corpus: List[Dict[str, Any]] = []

    # ---------- Corpus ingestion ----------

    def ingest_corpus(self):
        print(f"\n[Ingestion] Building scholarly corpus for topic: {self.topic}")
        self.corpus = build_corpus(self.topic, max_papers=Config.MAX_PAPERS)
        print(f"[Ingestion] Collected {len(self.corpus)} analyzed papers.")

    # ---------- Agent & Task definitions ----------

    def create_agents(self) -> List[Agent]:
        """Create specialized research agents"""

        corpus_summary = json.dumps(
            [
                {
                    "title": p.get("title", p.get("metadata", {}).get("title", "")),
                    "core_contribution": p.get("core_contribution", ""),
                    "problem_statement": p.get("problem_statement", ""),
                }
                for p in self.corpus
            ],
            indent=2,
        )

        shared_context = f"""
You have access to a shared research corpus of analyzed papers for the topic:

  "{self.topic}"

Here is a high-level overview of the corpus (titles + core contributions):

{corpus_summary}

Whenever you reason, ground your statements in this corpus as much as possible.
Cite paper titles or brief descriptions when you use specific findings.
"""

        research_analyst = Agent(
            role="Research Analyst",
            goal=f"Understand and summarize the key research directions about {self.topic}",
            backstory=f"""You are an expert research analyst specialized in {self.topic}.
You have access to a pre-ingested corpus of papers, including sections, equations,
and tables. You identify key findings, methodologies, and gaps.""",
            llm=self.llm_model_name,
            verbose=True,
            allow_delegation=False,
            memory=False,
            system_message=shared_context,
        )

        critical_reviewer = Agent(
            role="Critical Reviewer",
            goal="Critically evaluate the summarized findings and point out limitations, biases, and weaknesses.",
            backstory="""You are a rigorous academic reviewer with sharp critical thinking skills.
You excel at identifying methodological flaws, questionable assumptions,
and potential biases in research. You push for scientific rigor and reproducibility.""",
            llm=self.llm_model_name,
            verbose=True,
            allow_delegation=False,
            memory=False,
            system_message=shared_context,
        )

        synthesis_specialist = Agent(
            role="Synthesis Specialist",
            goal="Synthesize information from multiple papers to generate novel insights, connections, and hypotheses.",
            backstory="""You are a creative research synthesizer who connects disparate ideas,
spots emerging patterns, and proposes new research directions.""",
            llm=self.llm_model_name,
            verbose=True,
            allow_delegation=False,
            memory=False,
            system_message=shared_context,
        )

        research_coordinator = Agent(
            role="Research Coordinator",
            goal="Coordinate the research process and compile a coherent, well-structured Collective Insight Report.",
            backstory="""You are an experienced research project manager who ensures that
all aspects of the research question are thoroughly investigated and clearly communicated.""",
            llm=self.llm_model_name,
            verbose=True,
            allow_delegation=True,
            memory=False,
            system_message=shared_context,
        )

        return [
            research_analyst,
            critical_reviewer,
            synthesis_specialist,
            research_coordinator,
        ]

    def create_tasks(self, agents: List[Agent]) -> List[Task]:
        """Create research tasks for the agents"""

        research_analyst, critical_reviewer, synthesis_specialist, research_coordinator = agents

        literature_review = Task(
            description=f"""
You are given a pre-analyzed corpus of research papers about:

  "{self.topic}"

Each paper includes sections, equations, tables, results, limitations, and future work.

Your job:
1. Group the papers into 3–6 coherent themes or approaches.
2. For each theme, summarize:
   - Representative papers and their core contributions
   - Typical methods and experimental setups
   - Key equations or models (describe intuitively)
   - Main quantitative results (no need for exact numbers unless important)
3. Explicitly list limitations and open problems per theme.

Output a structured markdown report with clear headings per theme.
""",
            agent=research_analyst,
            expected_output="Structured thematic literature review over the ingested corpus.",
        )

        critical_analysis = Task(
            description="""
Critically analyze the thematic literature review.

For each theme in the review:
1. Evaluate methodological rigor (are methods appropriate and robust?).
2. Identify hidden assumptions, biases, and threats to validity.
3. Compare competing approaches and explain when one dominates another.
4. Suggest methodological improvements or alternative designs.
5. Point out under-explored subtopics or datasets.

Be specific and cite theme headings / paper titles where possible.
""",
            agent=critical_reviewer,
            context=[literature_review],
            expected_output="Detailed critique of the themes and methods, with clear suggestions and identified gaps.",
        )

        synthesis_task = Task(
            description=f"""
Using the literature review and critical analysis, synthesize higher-level insights about:

  "{self.topic}"

Steps:
1. Identify cross-theme patterns and tensions (e.g., performance vs interpretability).
2. Propose 3–6 concrete, testable research hypotheses.
3. For each hypothesis, sketch:
   - Intuition
   - Potential experimental setup
   - Expected outcomes and what they would imply
4. Highlight promising combinations of methods from different themes.
5. Suggest at least one ambitious, speculative research direction.

Make this section creative but grounded in evidence from the corpus.
""",
            agent=synthesis_specialist,
            context=[literature_review, critical_analysis],
            expected_output="Synthesis report with hypotheses, cross-theme insights, and research directions.",
        )

        coordination_task = Task(
            description="""
Coordinate the overall research process and compile the final Collective Insight Report.

Steps:
1. Read the literature review, critical analysis, and synthesis.
2. Resolve any contradictions or major inconsistencies.
3. Produce a final report with:
   - Executive summary (plain language, 3–5 paragraphs)
   - Thematic overview of the field
   - Critical insights (strengths, weaknesses, gaps)
   - Proposed hypotheses and research agenda
   - Brief “How we reasoned” section, outlining the agent workflow
4. Ensure reasoning is traceable: whenever you make a strong claim,
   point back to themes or paper titles in earlier sections.

The report should be clear enough for a grad student to use as a starting point.
""",
            agent=research_coordinator,
            context=[literature_review, critical_analysis, synthesis_task],
            expected_output="Final Collective Insight Report with executive summary and agenda.",
        )

        return [literature_review, critical_analysis, synthesis_task, coordination_task]

    # ---------- Core execution ----------

    def run_research(self) -> Dict[str, Any]:
        """Execute ingestion + multi-agent research process (one episode)"""

        print(f"\n=== Starting multi-agent research on: {self.topic} ===")

        # 1) Build corpus
        self.ingest_corpus()

        # 2) Agents & tasks
        agents = self.create_agents()
        tasks = self.create_tasks(agents)

        # 3) Run Crew
        research_crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
        )

        result = research_crew.kickoff()

        # 4) Reasoning graph + interaction log
        self._build_reasoning_graph(agents, tasks)
        self._log_interactions(agents, tasks)

        return {
            "topic": self.topic,
            "result": str(result),
            "timestamp": datetime.now().isoformat(),
            "reasoning_graph": self.reasoning_graph,
            "conversation_history": self.conversation_history,
            "agent_interactions": self.agent_interactions,
            "corpus": self.corpus,
        }

    # ---------- Helpers for RL / visualization ----------

    def _build_reasoning_graph(self, agents: List[Agent], tasks: List[Task]):
        for agent in agents:
            self.reasoning_graph.add_node(agent.role, type="agent")

        for task in tasks:
            label = task.description.strip().splitlines()[0][:60] + "..."
            self.reasoning_graph.add_node(label, type="task")

            if hasattr(task, "agent") and task.agent:
                self.reasoning_graph.add_edge(task.agent.role, label)

            if hasattr(task, "context") and task.context:
                task_context = task.context if isinstance(task.context,list) else []
                for ctx in task_context:
                    ctx_label = ctx.description.strip().splitlines()[0][:60] + "..."
                    self.reasoning_graph.add_edge(ctx_label, label)

    def _log_interactions(self, agents: List[Agent], tasks: List[Task]):
        self.agent_interactions = []
        for t in tasks:
            role = t.agent.role if hasattr(t, "agent") else "Unknown"

            if "literature" in t.description.lower():
                ttype = "literature"
            elif "critical" in t.description.lower():
                ttype = "critique"
            elif "synthesize" in t.description.lower() or "synthesis" in t.description.lower():
                ttype = "synthesis"
            else:
                ttype = "coordination"

            approach = "exploratory" if role in ["Research Analyst", "Synthesis Specialist"] else "conservative"

            self.agent_interactions.append(
                {
                    "agent_role": role,
                    "task_type": ttype,
                    "approach": approach,
                }
            )


# ---------- Streamlit dashboard (optional) ----------

import streamlit as st


def plot_reasoning_graph(graph: nx.DiGraph):
    pos = nx.spring_layout(graph)

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(1 if graph.nodes[node]["type"] == "agent" else 0)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        marker=dict(size=20, color=node_color, colorscale="Viridis", line=dict(width=2)),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Research Reasoning Flow",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


def create_dashboard(research_system: ResearchAgentSystem, results: Dict[str, Any]):
    st.title("🤖 Agentic AI Research Lab (Gemini + MARL)")
    st.subheader(f"Research Topic: {research_system.topic}")

    st.header("Best Episode: Research Results")
    st.write(results["result"])

    st.header("Research Process Flow")
    fig = plot_reasoning_graph(research_system.reasoning_graph)
    st.plotly_chart(fig)

    st.header("Agent Interactions (used for RL reward)")
    st.json(results.get("agent_interactions", []))

    st.header("Corpus (truncated)")
    st.json(results.get("corpus", [])[:3])
