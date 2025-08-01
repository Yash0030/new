from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
import os
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    do_sample=False,
    repetition_penalty=1.0,
)

model=ChatHuggingFace(llm=llm)

# Define output parser to return JSON
parser = JsonOutputParser()

# LangChain Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an intelligent assistant that expands only vague insurance or policy-related queries 
     and provides a reasoning path for document retrieval and decision making.
     
     Given a natural language query, do one thing:
     1. Extract these (age, location, treatment, policy duration) information only if given and  Expand the query with relevant structure 
    
     
     Return  
     - "expanded_query" (string)
    """),
     
    ("human", "{query}")
])

# Chain: Prompt → LLM → JSON Parser
query_expansion_chain: Runnable = prompt | model | parser
 # 2. Generate a chain of reasoning steps that explain how a decision could be made using policy clauses.
# - "thought_steps" (string with bullet points or step-by-step reasoning)
def expand_query_and_thought(query: str) -> dict:
    """
    Run the LangChain query expansion
    :param query: user input string
    :return: expanded_query 
    """
    try:
        return query_expansion_chain.invoke({"query": query})
    except Exception as e:
        return {
            "expanded_query": query
        }
