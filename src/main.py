import asyncio
import threading
import sys
import json 
from datetime import datetime, timezone

from LiveTranscriber import LiveTranscriber
import ollama
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.utils.maintenance.graph_data_operations import clear_data # Kept if clearing data is a potential operation
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
import pyttsx3

neo4j_uri = "neo4j://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "qualcomm"


llm_config = LLMConfig( 
    api_key="abc",
    model="qwen2.5:1.5b", # Model for Graphiti's internal LLM client
    small_model="qwen2.5:1.5b",
    base_url="http://localhost:11434/v1",
)
llm_client = OpenAIClient(config=llm_config)

graphiti = Graphiti(
    os.environ.get("NEO4J_URI") or "neo4j://localhost:7687",
    os.environ.get("NEO4J_USER") or "neo4j",
    os.environ.get("NEO4J_PASSWORD") or "qualcomm",
    llm_client=llm_client,
    embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="abc",
            embedding_model="nomic-embed-text:latest",
            embedding_dim=4096,
            base_url="http://localhost:11434/v1",
        )
    ),
    cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
    )
    
class GraphitiSearchResult(BaseModel):
    uuid: str = Field(description="Unique identifier for the episode")
    fact: str = Field(description="The factual statement recieved from knowledge graph")
    valid_at: Optional[str] = Field(
        default=None, description="The time when the fact was valid (if known)"
    )
    invalid_at: Optional[str] = Field(
        default=None, description="The time when the fact was invalid (if known)"
    )
    source_node_uuid: Optional[str] = Field(
        default=None, description="The UUID of the source node from which the fact was derived"
    )

class EpisodeFormat(BaseModel):
    content: str = Field(description="The content of the episode")
    type: EpisodeType = Field(
        default=EpisodeType.text, description="The type of the episode"
    )
    description: Optional[str] = Field(
        default=None, description="A description of the episode"
    )

episode_counter = 0 # Global counter for episodes

async def add_episodes(content: str, ep_description: str):
    """
    Adds a new episode to the knowledge graph.
    Use this tool to store a piece of information or a thing to remember.
    Provide 'content' (the information itself) and 'ep_description' (a brief description of the information type or context).
    Example: If the user says "I will go to Bangalore tomorrow", invoke with: {"content": "You are going to Bangalore tomorrow", "ep_description": "User's destination and place of travel"}
    """
    global episode_counter 
    episode_counter += 1
    prefix = "Episode"
    await graphiti.add_episode( 
        episode_body=content,
        source=EpisodeType.text,
        name=f"{prefix} {episode_counter}",
        reference_time=datetime.now(timezone.utc),
        source_description=ep_description
    )
    print(f"Added episode: {prefix} {episode_counter} with content: \"{content}\", description: \"{ep_description}\"")
    return f"Successfully added episode: {prefix} {episode_counter} with content: \"{content}\", description: \"{ep_description}\""


async def search_episodes(query: str) -> list[GraphitiSearchResult]:
    """
    Searches the knowledge graph for episodes relevant to a question.
    Use this tool to find or retrieve information.
    Provide only the 'query' argument as a string.
    Example: If the user asks "Where are my keys?", invoke with: {"query": "Where are my keys?"}
    """
    try:
        results = await graphiti.search(query=query) 
        formatted_results = []
        for result in results:
            formatted_result = GraphitiSearchResult(
                uuid=result.uuid,
                fact=result.fact,
                source_node_uuid=result.source_node_uuid if result.source_node_uuid else None,
            )
            if hasattr(result, 'valid_at') and result.valid_at:
                formatted_result.valid_at = str(result.valid_at)
            if hasattr(result, 'invalid_at') and result.invalid_at:
                formatted_result.invalid_at = str(result.invalid_at)
            formatted_results.append(formatted_result)
        
        if not formatted_results: # Check if results list is empty
            print(f"Search for \"{query}\" returned no results.")
            return None

        print(f"Search for \"{query}\" returned {formatted_results[0].fact} results.")
        
        query_keywords = {kw.lower() for kw in query.split() if kw.strip()}
        first_fact_lower = formatted_results[0].fact.lower()
        
        # Check if any keyword from the query is present in the first fact.
        is_relevant = any(keyword in first_fact_lower for keyword in query_keywords) if query_keywords else False
        
        if is_relevant:
            return formatted_results
        else:
            # The result was found but deemed not relevant by keyword check.
            print(f"Info: The result for query \"{query}\" (fact: \"{formatted_results[0].fact}\") is not being returned as it lacks keyword relevance.")
            return None
    except Exception as e:
        print(f"Error during search for \"{query}\": {e}")
        return [{"error": f"Error during search: {str(e)}"}]

def get_ollama_tool_schemas() -> list: 
    """
    Returns the schemas for tools in the format expected by ollama.chat.
    """
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'add_episodes',
                'description': (
                    "Adds a new episode to the knowledge graph. Use this tool to store a piece of information. "
                    "Provide 'content' (the information itself) and 'ep_description' (a brief description of the information type or context). "
                    "Example: If the user says \"My keys are on the table\", invoke with: {\"content\": \"My keys are on the table\", \"ep_description\": \"User's keys location\"}. "
                    "Another example: If the user says \"Teacher said chapter 5 of physics book is important for exam\", "
                    "invoke with: {\"content\": \"Teacher said chapter 5 of physics book is important for exam\", \"ep_description\": \"Exam advice from teacher\"}"
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'content': {'type': 'string', 'description': 'The information to store as an episode.'},
                        'ep_description': {'type': 'string', 'description': 'A brief description of the information or its context.'}
                    },
                    'required': ['content', 'ep_description']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'search_episodes',
                'description': (
                    "Searches the knowledge graph for episodes whenever a question was asked as a query. Use this tool to find or retrieve information. "
                    "Provide only the 'query' argument as a string. Example: If the user asks \"Where are my keys?\", invoke with: {\"query\": \"Where are my keys?\"}"
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'The query to search for in the knowledge graph.'}
                    },
                    'required': ['query']
                }
            }
        }
    ]
    return tools

# Global variables
event_loop = None
# graphiti_instance = None # REMOVED - using global 'graphiti' directly
ollama_model_name = "qwen2.5:1.5b"

def handle_new_transcript(transcript: str):
    if transcript and event_loop:
        print(f"Main: Received transcript: \"{transcript}\"")
        
        async def process_transcript_with_agent():
            nonlocal transcript
            try:
                print(f"KG Agent: Processing fact: \"{transcript}\"")
                
                system_prompt = (
                    "You are an assistant that can store information or search for it in a knowledge graph. "
                    "When the user provides a statement, decide whether to use the 'add_episodes' tool to store it, "
                    "or the 'search_episodes' tool to retrieve information. "
                    "Only use the provided tools. Ensure you provide all required arguments for the chosen tool. "
                    "Whenever a new fact regardless whether it was associated with the same entity is provided, use the 'add_episodes' tool to store it. "
                    "Decide whether not to use a tool based on the content of the transcript. "
                    "Do not respond with conversational text unless no tool is called."
                )
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': transcript}
                ]
                
                tool_schemas = get_ollama_tool_schemas()
                
                response = await ollama.AsyncClient().chat(
                    model=ollama_model_name,
                    messages=messages,
                    tools=tool_schemas
                )
                # print(f"KG Agent: Ollama Raw Response: {response}") # For debugging

                available_functions = {
                    "add_episodes": add_episodes,
                    "search_episodes": search_episodes,
                }
                
                if response.message.tool_calls:
                    for tool in response.message.tool_calls:
                        function_to_call = available_functions.get(tool.function.name)
                        if function_to_call:
                            results = await function_to_call(**tool.function.arguments)
                            if tool.function.name == "search_episodes":
                                if results and isinstance(results, list) and results[0] and hasattr(results[0], 'fact'): # Ensure results are valid
                                    engine = pyttsx3.init()
                                    engine.setProperty('rate', 150)
                                    engine.say(results[0].fact) 
                                    engine.runAndWait()
                                elif results and isinstance(results, list) and results[0] and "error" in results[0]:
                                    print(f"Search tool returned an error: {results[0]['error']}")
                                else:
                                    print(f"Search tool did not return a speakable fact for query: {tool.function.arguments.get('query')}")
                        else:
                            print(f'Function not found: {tool.function.name}')
                else:
                    # This block executes if Ollama decides not to call any tool.
                    # You might want to log this or handle it specifically.
                    print(f"KG Agent: Ollama decided not to call any tool for transcript: \"{transcript}\"")
                    # If you want the agent to respond conversationally when no tool is called:
                    # if response.message.content:
                    #     print(f"KG Agent: Ollama response: {response.message.content}")
                    #     engine = pyttsx3.init()
                    #     engine.setProperty('rate', 150)
                    #     engine.say(response.message.content)
                    #     engine.runAndWait()


            except Exception as e:
                print(f"Error processing transcript with agent: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

        asyncio.run_coroutine_threadsafe(process_transcript_with_agent(), event_loop)
    elif not transcript:
        # Handles cases where an empty transcript might be received.
        pass 

async def main_async_runner(graphiti):
    global event_loop
    event_loop = asyncio.get_running_loop()

    print("Main: Initializing Knowledge Graph...")
    try:
        await graphiti.build_indices_and_constraints() 
        # await clear_data(graphiti.driver) # Uncomment to clear existing data on startup
        print("Main: Knowledge Graph initialized.")
    except Exception as e:
        print(f"Failed to initialize Knowledge Graph: {e}", file=sys.stderr)
        return

    print("Main: Initializing Live Transcriber...")
    transcriber = LiveTranscriber(transcript_callback=handle_new_transcript)
    print("Main: Live Transcriber initialized.")

    transcriber_thread = threading.Thread(target=transcriber.run, daemon=True)
    transcriber_thread.start()
    print("Main: Live Transcriber started in a separate thread. Press Ctrl+C to stop.")

    try:
        while transcriber_thread.is_alive():
            await asyncio.sleep(0.5) 
    except KeyboardInterrupt:
        print("\nMain: Keyboard interrupt received. Initiating shutdown...")
    finally:
        print("Main: Stopping Live Transcriber...")
        if 'transcriber' in locals() and transcriber:
            transcriber.stop_event.set() 
        if 'transcriber_thread' in locals() and transcriber_thread.is_alive():
            transcriber_thread.join(timeout=5) 
        
        # graphiti_core.Graphiti doesn't have an explicit async close method in typical usage.
        # Connections are usually managed by the underlying Neo4j driver.
        print("Main: Shutdown complete.")

if __name__ == "__main__":
    import tkinter as tk  # GUI for manual input and control

    def start_listening():
        """Starts the main asynchronous process in a new thread."""
        threading.Thread(target=lambda: asyncio.run(main_async_runner(graphiti)), daemon=True).start()

    def remember_memory():
        """Adds a memory to the knowledge graph based on GUI input."""
        content = content_entry.get()
        description = desc_entry.get()
        if content and description:
            # Run in the existing event loop if available, or create a new one.
            # This is a simplification; for robust GUI + asyncio, a dedicated library might be better.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(add_episodes(content, description), loop)
                else:
                    asyncio.run(add_episodes(content, description))
            except RuntimeError: # No current event loop
                 asyncio.run(add_episodes(content, description))
            content_entry.delete(0, tk.END)
            desc_entry.delete(0, tk.END)
        else:
            print("Content and description cannot be empty.")


    root = tk.Tk()
    root.title("MindGuardAI Companion")
    tk.Label(root, text="Memory Content:").pack(pady=(10,0))
    content_entry = tk.Entry(root, width=50)
    content_entry.pack(pady=5, padx=10)
    tk.Label(root, text="Description (e.g., 'Location of keys', 'Meeting reminder'):").pack(pady=(10,0))
    desc_entry = tk.Entry(root, width=50)
    desc_entry.pack(pady=5, padx=10)
    tk.Button(root, text="Remember Memory", command=remember_memory).pack(pady=(10,5))
    tk.Button(root, text="Start Listening", command=start_listening).pack(pady=(0,10))
    
    # Instructions label
    instructions = (
        "Instructions:\n"
        "1. Click 'Start Listening' to activate voice transcription.\n"
        "2. Speak clearly. Your speech will be processed to store or retrieve information.\n"
        "   - To store: Make a statement (e.g., 'My flight to London is on Tuesday').\n"
        "   - To retrieve: Ask a question (e.g., 'When is my flight to London?').\n"
        "3. Alternatively, use the 'Memory Content' and 'Description' fields to manually add information, then click 'Remember Memory'.\n"
        "4. Close this window or press Ctrl+C in the console (if visible) to stop."
    )
    tk.Label(root, text=instructions, justify=tk.LEFT, wraplength=380).pack(pady=(10,10), padx=10)

    root.mainloop()