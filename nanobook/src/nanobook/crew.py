from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class Nanobook():
    """Nanobook crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def documentation_extraction_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['documentation_extraction_specialist'], # type: ignore[index]
            verbose=True
        )

    @task
    def documentation_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['documentation_extraction_task'], # type: ignore[index]
            agent=self.documentation_extraction_specialist()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Nanobook crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
