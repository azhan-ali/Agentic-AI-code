import sys
import os 
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

# Ensure UTF-8 output for Windows console (to print emojis like 🔏 and 🌍)
if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Using ChatGroq as Mistral API free tier has 0 req/min quota limit (HTTP 429)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)
def merge_score_dicts(existing : dict , newupdate:dict) -> dict:
    if existing is None:
        return newupdate
    return {**existing,**newupdate}

## create a state 
class AnalyzeState(TypedDict):
    raw_text : str
    safety_score : Annotated[dict[str,int],merge_score_dicts]

## nodes 

# Toxicity node 

def toxicity_node(state: AnalyzeState) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response= llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_score" : {"toxicity_level" : score}}

## copyright node 
def copyright_node(state: AnalyzeState) -> dict:
    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_score": {"copyright_risk": score}}

## cultral node
def culture_node(state: AnalyzeState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_score": {"cultural_sensitivity": score}}

## now lets create the graph 
builder = StateGraph(AnalyzeState)

# adding node 
builder.add_node("toxicity",toxicity_node)
builder.add_node("copyright",copyright_node)
builder.add_node("culture",culture_node)

## now lets make edges
builder.add_edge(START,"toxicity")
builder.add_edge(START,"copyright")
builder.add_edge(START,"culture")

builder.add_edge("toxicity",END)
builder.add_edge("copyright",END)
builder.add_edge("culture",END)

app = builder.compile()


sample_script = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """

initial_state = {
    "raw_text" : sample_script,
    "safety_score" : {} ## initialized as an empty dictionary
}

final_state = app.invoke(initial_state)

print(final_state["safety_score"])