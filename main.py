# main.py
import os

import os
from dotenv import load_dotenv

load_dotenv()

# Force-set environment vars for CrewAI LLM initialization
os.environ["GOOGLE_API_KEY"] = "AIzaSyAHyTYc8Hw3uOVI_uOSA-UnQv9OffsPnmw"

# Disable native provider so we don't hit the error path again
os.environ["CREWAI_USE_NATIVE_GEMINI"] = "false"


# Optional: suppress gRPC / GLOG noise
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from research_agent_rl import EnhancedResearchAgentSystem


def main():
    research_topic = (
        "Multi-agent reinforcement learning for traffic signal control in urban networks"
    )

    num_episodes = 3

    system = EnhancedResearchAgentSystem(topic=research_topic, num_agents=4)
    best_results = system.run_enhanced_research(num_episodes=num_episodes)

    print("\n" + "=" * 60)
    print("BEST EPISODE – COLLECTIVE INSIGHT REPORT")
    print("=" * 60)
    print(best_results["result"])
    print(f"\nBest collaboration score: {best_results['best_collaboration_score']:.3f}")

    os.makedirs("output", exist_ok=True)
    report_path = os.path.join("output", "collective_insight_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Collective Insight Report\n\n")
        f.write(f"## Topic\n\n{research_topic}\n\n")
        f.write(best_results["result"])

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
