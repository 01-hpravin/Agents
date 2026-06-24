from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import (
    ScrapeWebsiteTool,
    SerperDevTool,
    FileWriterTool,
    FileReadTool,
    MDXSearchTool,
    )

# -------------------------------------------------
# Tool Initialization
# -------------------------------------------------

website_scrape_tool = ScrapeWebsiteTool()
search_tool = SerperDevTool()

file_writer_tool = FileWriterTool()
file_read_tool = FileReadTool()

mdx_search_tool = MDXSearchTool()

@CrewBase
class Nanobook():
    """Nanobook crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def documentation_extraction_specialist(self) -> Agent:
        """
        Scrapes official docs and READMEs, strips noise, and produces a clean
        Markdown knowledge base.  Needs web access and search.
        """
        return Agent(
            config=self.agents_config['documentation_extraction_specialist'],  # type: ignore[index]
            verbose=True,
            tools=[
                search_tool,        # find the right URLs to scrape
                website_scrape_tool,  # pull full page content
            ],
        )

    @agent
    def technical_architect_analyst(self) -> Agent:
        """
        Reads the extracted docs and builds a Concept Map.  Uses semantic
        search to quickly re-query the extraction output without re-reading it
        in full.
        """
        return Agent(
            config=self.agents_config['technical_architect_analyst'],  # type: ignore[index]
            verbose=True,
            tools=[
                mdx_search_tool,  # semantic search over extracted Markdown
            ],
        )
 
    @agent
    def curriculum_designer(self) -> Agent:
        """
        Designs the Table of Contents and per-chapter briefs.  Uses semantic
        search to cross-reference the Concept Map while planning.
        """
        return Agent(
            config=self.agents_config['curriculum_designer'],  # type: ignore[index]
            verbose=True,
            tools=[
                mdx_search_tool,  # locate relevant concepts quickly
            ],
        )
    
    @agent
    def technical_instructional_writer(self) -> Agent:
        """
        Writes the full NanoBook prose chapter by chapter, then saves the
        draft to disk so the QA Editor can review it.
        """
        return Agent(
            config=self.agents_config['technical_instructional_writer'],  # type: ignore[index]
            verbose=True,
            tools=[
                mdx_search_tool,   # look up facts from extracted docs mid-writing
                file_writer_tool,  # persist the draft to nanobook_draft.md
            ],
        )
 
    @agent
    def pedagogical_quality_assurance_editor(self) -> Agent:
        """
        Reads the draft, performs gate-checks, and either returns a rejection
        report or delivers the final polished NanoBook.
        """
        return Agent(
            config=self.agents_config['pedagogical_quality_assurance_editor'],  # type: ignore[index]
            verbose=True,
            tools=[
                file_read_tool,   # read the saved draft from disk
                mdx_search_tool,  # cross-reference claims against extracted docs
                file_writer_tool, # write the final approved NanoBook to disk
            ],
        )
    

    # -----------------------------------------------------------------------
    # Tasks
    # -----------------------------------------------------------------------
 
    @task
    def documentation_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['documentation_extraction_task'],  # type: ignore[index]
            agent=self.documentation_extraction_specialist(),
        )
 
    @task
    def concept_mapping_task(self) -> Task:
        return Task(
            config=self.tasks_config['concept_mapping_task'],  # type: ignore[index]
            agent=self.technical_architect_analyst(),
            context=[self.documentation_extraction_task()],
        )
 
    @task
    def curriculum_design_task(self) -> Task:
        return Task(
            config=self.tasks_config['curriculum_design_task'],  # type: ignore[index]
            agent=self.curriculum_designer(),
            context=[
                self.documentation_extraction_task(),
                self.concept_mapping_task(),
            ],
        )
 
    @task
    def content_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['content_writing_task'],  # type: ignore[index]
            agent=self.technical_instructional_writer(),
            context=[
                self.documentation_extraction_task(),
                self.concept_mapping_task(),
                self.curriculum_design_task(),
            ],
        )
 
    @task
    def quality_assurance_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_task'],  # type: ignore[index]
            agent=self.pedagogical_quality_assurance_editor(),
            context=[
                self.documentation_extraction_task(),
                self.curriculum_design_task(),
                self.content_writing_task(),
            ],
            output_file='nanobook_output.md',
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
