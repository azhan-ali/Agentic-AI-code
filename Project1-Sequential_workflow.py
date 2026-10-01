import sys
import os 
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Ensure UTF-8 output for Windows console
if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8')

## lets create the state first 
class pipelinestate(TypedDict):
    raw_input : str
    edited_text : str
    script_text : str
    final_output : str

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)


## node is just a python function 
## lets make the node

## editor node 
def editor_node(state : pipelinestate)-> dict:
    """Stage1 : cleans up grammar , remove typos and refines the tone """
    print("\n--- [Stage 1] Executing Editor Node")
    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow "
        "while keeping the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {"edited_text" : response.content.strip()}

## Scriptwriter node
def sciptwriter_node(state : pipelinestate)-> dict:
    """Stage2 : Formats the clean text into an engaging video script style. """
    print("\n--- [Stage 2] Executing scripwriter Node")
    prompt = (
        """
        You are an expert scriptwriter.
        Turn the edited content into an engaging, clear, and natural script while preserving its meaning.
        Improve the flow and storytelling, keep it concise, and output only the final script.
        """
        f"Edited text : \n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {"script_text" : response.content.strip()}

## Translator node 
def translator_node(state : pipelinestate)-> dict:
    """Stage3 : Translates the script into natural flowing hinglish. """
    print("\n--- [Stage 3] Executing Hinglish Translator Node")
    prompt = (
        """
        You are an expert Hinglish content writer.
        Convert the script into natural, conversational Hinglish using Roman English, 
        while preserving its meaning and flow.
        Keep it engaging, easy to understand, and output only the final Hinglish script.
        """
        f"script : \n{state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {'final_output' : response.content.strip()}


## now our state andnodes are ready and now it is time to create the graph
# and for creating the graph we have to connect these nodes and for connecting these node 
# we have to use the edges 
# edges are very important to create the workflows

from langgraph.graph import StateGraph,END,START

## create the graph 
graph = StateGraph(pipelinestate)

##  add the nodes
graph.add_node('editor',editor_node)
graph.add_node('scriptwriter',sciptwriter_node)
graph.add_node('translator',translator_node)

## now connect the nodes
graph.add_edge(START,'editor')
graph.add_edge('editor','scriptwriter')
graph.add_edge('scriptwriter','translator')
graph.add_edge('translator',END)

## now compile the graph 
workflow = graph.compile()

## workflow is runnable so invoke kr sakte hai 

result = workflow.invoke({
    'raw_input' : "The sun rises in the east, and sets in the west."
    })

print(result['final_output'])