import os
import uuid
import yaml
import json
from pathlib import Path

from pydantic import BaseModel, Field
from typing import List, Optional

from crewai import LLM, Agent, Task, Crew
from crewai_tools import DirectoryReadTool, FileReadTool
from crewai.flow.flow import Flow, start, listen, router, or_

from firecrawl import FirecrawlApp


# Define LLM
llm = LLM(
    model="gpt-4o-mini",
    max_tokens=1000,  # Limit response tokens
)

with open("config/planner_agents.yaml", "r") as f:
    agents_config = yaml.safe_load(f)

with open("config/planner_tasks.yaml", "r") as f:
    tasks_config = yaml.safe_load(f)

class Tweet(BaseModel):
    """
    Represents an individual tweet in a thread
    """
    content: str
    is_hook: bool
    media_urls: Optional[List[str]] = []

class Thread(BaseModel):
    """
    Represents a Twitter thread
    """
    topic: str
    tweets: list[Tweet]    

class LinkedInPost(BaseModel):
    """
    Represents a LinkedIn post
    """
    content: str
    media_url: str


all_tools = [DirectoryReadTool(), FileReadTool()]


draft_analyzer = Agent(
    config=agents_config["draft_analyzer"],
    tools=all_tools,
    llm = llm
)

analyze_draft = Task(
    config=tasks_config["analyze_draft"],
    agent=draft_analyzer
)

twitter_thread_planner = Agent(
    config=agents_config['twitter_thread_planner'],
    tools=all_tools,
    llm=llm
    )
create_twitter_thread_plan = Task(config=tasks_config['create_twitter_thread_plan'],
    agent=twitter_thread_planner,
    output_pydantic=Thread
    )    

linkedin_post_planner = Agent(
    config=agents_config['linkedin_post_planner'],
    tools=all_tools,
    llm=llm
)

create_linkedin_post_plan = Task(
    config=tasks_config['create_linkedin_post_plan'],
    agent=linkedin_post_planner,
    output_pydantic=LinkedInPost
)

# Define Crews
twitter_planning_crew = Crew(
    agents=[draft_analyzer, twitter_thread_planner],
    tasks=[analyze_draft, create_twitter_thread_plan],
    verbose=True
)

linkedin_planning_crew = Crew(
    agents=[draft_analyzer, linkedin_post_planner],
    tasks=[analyze_draft, create_linkedin_post_plan],
    verbose=True
)

class ContentPlanningState(BaseModel):
    """
    State for the content planning flow
    """
    # URL of the blog to scrape
    blog_post_url: str = Field(default="https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag")

    # Path where the scrapped content will be stored
    draft_path: Path = "asset/"

    # Determines whether to create a Twitter or LinkedIn post 
    post_type: str = "twitter"  
    
    # Example Twitter threads for style reference
    path_to_example_threads: str = "assets/example_threads.txt" 
    
    # Example LinkedIn posts for reference
    path_to_example_linkedin: str = "assets/example_linkedin.txt"

# Helper function to truncate large files
def truncate_file_content(file_path, max_chars=50000):
    """Truncate a file to a maximum number of characters"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    if len(content) > max_chars:
        truncated_content = content[:max_chars] + "\n\n[Content truncated due to length...]"
        with open(file_path, 'w') as f:
            f.write(truncated_content)
        print(f"File truncated from {len(content)} to {max_chars} characters")
    
    return file_path

class CreateContentPlanningFlow(Flow[ContentPlanningState]):

    @start()
    def scrape_blog_post(self):
        print(f"# Fetching draft from: {self.state.blog_post_url}")

        # Initialize FireCrawl
        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        
        # Scrape the blog post in Markdown and HTML format
        scrape_result = app.scrape_url(
            self.state.blog_post_url,
            formats=['markdown', 'html']
        )

        # Extract the title (fallback to a UUID if not found)
        try:
            title = scrape_result.metadata.title
        except Exception:
            title = str(uuid.uuid4())

        # Store the scraped content as a markdown file
        self.state.draft_path = f'assets/{title}.md'
        with open(self.state.draft_path, 'w') as f:
            f.write(scrape_result.markdown)
        
        # Truncate the file to avoid context window issues
        self.state.draft_path = truncate_file_content(self.state.draft_path)

        return self.state
    
    @router(scrape_blog_post)
    def select_platform(self):
        if self.state.post_type == "twitter":
            return "twitter"
        elif self.state.post_type == "linkedin":
            return "linkedin"
    
    @listen("twitter")
    def twitter_draft(self):
        print(f"# Planning content for: {self.state.draft_path}")
    
        # Execute the Twitter Planning Crew
        result = twitter_planning_crew.kickoff(inputs={
            'draft_path': self.state.draft_path, 
            'path_to_example_threads': self.state.path_to_example_threads
        })
    
        print(f"# Planned content for {self.state.draft_path}:")
    
        # Print each tweet in the generated thread
        for i, tweet in enumerate(result.pydantic.tweets):
            print(f"Tweet {i+1}: {tweet.content}")
            print(f"Media URLs: {tweet.media_urls}")
            print("-" * 100)
    
        return result
    
    @listen("linkedin")
    def linkedin_draft(self):
        print(f"# Planning content for: {self.state.draft_path}")
    
        # Execute the LinkedIn Planning Crew
        result = linkedin_planning_crew.kickoff(inputs={
            'draft_path': self.state.draft_path, 
            'path_to_example_linkedin': self.state.path_to_example_linkedin
        })
    
        print(f"# Planned content for {self.state.draft_path}:")
        print(f"{result.pydantic.content}")
    
        return result
    
    @listen(or_(twitter_draft, linkedin_draft))
    def save_plan(self, plan):
        with open(f'output/draft.json', 'w') as f:
            json.dump(plan.pydantic.model_dump(), f, indent=2)


if __name__ == "__main__":
    blog_post_url = "https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag"
    draft_path = "assets/"
    post_type = "twitter"
    path_to_example_threads = "assets/example_threads.txt"
    path_to_example_linkedin = "assets/example_linkedin.txt"
    
    # Truncate example files to avoid context window issues
    truncate_file_content(path_to_example_threads)
    truncate_file_content(path_to_example_linkedin)
    
    flow = CreateContentPlanningFlow()
    flow.kickoff(inputs={
        "blog_post_url": blog_post_url,
        "draft_path": draft_path,
        "post_type": post_type,
        "path_to_example_threads": path_to_example_threads,
        "path_to_example_linkedin": path_to_example_linkedin
    })