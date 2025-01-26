import neo4j
from neo4j import GraphDatabase, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends, responses, security, BackgroundTasks, Form
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from typing import Dict, List
from datetime import datetime
import random
import ast
from fastapi import APIRouter, HTTPException
from pathlib import Path
from openai import OpenAI


# In-memory storage for chat histories
chat_histories: Dict[str, Dict[str, List[str]]] = {}

# Create FastAPI instance
router = APIRouter()

class Settings(BaseModel):
    language: str
    politeness_level: str
    formality: str
    creativity: float
    response_length: str

class Question(BaseModel):
    question: str
    session_id: str
    settings: Settings
    time: str = None
    date: str = None
    question_history: List[str] = []
    answer_history: List[str] = []


# Load environment variables
load_dotenv()

# Get environment variables
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
openai_api_key = os.getenv("OPENAI_API_KEY")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-3.5-turbo")

# Function to connect to Neo4j
def connect_to_neo4j():
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password)) # Connect to the Neo4j database
        driver.verify_connectivity() # Verify the connection
        print("Connection to Neo4j established successfully!")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {str(e)}") # Print the error message
        return None

# Connect to Neo4j
driver = connect_to_neo4j()

# Load JSON data
with open('routers/sample.json', 'r') as file:
    data = json.load(file)

# Extract questions and their corresponding cypher queries
questions = [item['question'] for item in data['data']]
cypher_queries = [item['cypher_query'] for item in data['data']]


# Function to find the top 5 similar questions
def find_similar_questions(user_query, questions, cypher_queries, top_n=5):
    # Combine user query with the questions
    corpus = questions + [user_query]

    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Compute cosine similarity of user query with questions
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Get the top N similar questions
    similar_indices = similarities.argsort()[-top_n:][::-1]

    # Return the question, cypher_query, and similarity score
    similar_questions = [
        (questions[i], cypher_queries[i], similarities[i]) for i in similar_indices
    ]

    return similar_questions

def generate_query(user_question: str, session_id: str, question: Question) -> str:
    recent_history = []  # Initialize an empty list to store recent conversation history
    if session_id in chat_histories:  # Check if the session ID exists in the chat histories
        questions = chat_histories[session_id]["questions"]  # Get the questions from the chat histories
        answers = chat_histories[session_id]["answers"]  # Get the answers from the chat histories
        # Get the last 3 items, excluding the current question
        for i in range(max(len(questions) - 4, 0), len(questions) - 1):
            recent_history.append({
                "question": questions[i],
                "answer": answers[i]
            })

    # Format the conversation history
    history_text = "No previous conversation." if not recent_history else "\n".join([
        f"Previous Question: {item['question']}\n"
        f"Previous Answer: {item['answer']}\n"
        for item in recent_history
    ])

    print(history_text)

    # Define template as a constant
    TEMPLATE_CYPHER = {
        "text_explanation": "",
        "query_generation_status": "Yes/No",
        "query": "OPTIONAL MATCH query here",
        "plotly_code_generation_status": "Yes/No",
        "plotly_code": "plotly code"
    }

    # Find the top 5 similar questions
    settings_prompt = f"""
       Response Settings:
       - Language: {question.settings.language}
       - Politeness Level: {question.settings.politeness_level}
       - Formality: {question.settings.formality}
       - Response Length: {question.settings.response_length}
       - Current date: {question.date}
       - Current time: {question.time}

       Please adjust your response according to these settings:
       1. Use the specified {question.settings.language} language
       2. Maintain {question.settings.politeness_level} politeness level
       3. Keep {question.settings.formality} tone
       5. Provide {question.settings.response_length} response length
       """

    # Move the prompt template outside the function
    QUERY_CHAT_PROMPT = """Role and Context: """

    retry_count = 0  # Initialize retry count
    max_retries = 4  # Set maximum number of retries

    try:
        similar_results = find_similar_questions(user_question, questions, cypher_queries)
        similar_text = "\n".join([
            f"Reference Q: {q}\nReference Query: {c}\nSimilarity: {s:.2f}"
            for q, c, s in similar_results
        ])
    except Exception as e:
        similar_text = "No similar examples found."


    # Retry loop for API call
    while retry_count < max_retries:
        try:
            # Construct chat messages
            chat_messages = [
                {
                    "role": "system",  # System role for the AI assistant
                    "content": [  # Content of the message
                        {
                            "type": "text",
                            "text": QUERY_CHAT_PROMPT  # Add the formatted prompt to the content
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_question  # Add the user question to the content
                        }
                    ]
                }
            ]

            client = OpenAI(api_key=openai_api_key)  # Initialize the OpenAI client

            # Make API call
            completion = client.chat.completions.create(
                model=deployment,
                messages=chat_messages,
                max_tokens=4096,
                temperature=question.settings.creativity,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )

            # Check if the response contains the query generation status
            return completion.choices[0].message.content

        # Retry on API errors
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return f"Error after {max_retries} retries: {str(e)}"
            continue
    # Return error message if maximum retries exceeded
    return "Maximum retries exceeded"


# Define the FastAPI router
@router.post("/start_session")
async def start_session():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    chat_histories[session_id] = {
        "questions": [],
        "answers": []
    }
    return JSONResponse(content={"session_id": session_id})  # Return the session ID as JSON


# Define the FastAPI router
def serialize_neo4j_value(val):
    """Serialize Neo4j specific data types"""
    if isinstance(val, (neo4j.time.DateTime, neo4j.time.Date)): # Check for DateTime or Date objects
        return val.iso_format()
    elif isinstance(val, neo4j.time.Time): # Check for Time objects
        return str(val)
    elif isinstance(val, (int, float, str, bool)): # Check for basic data types
        return val
    elif val is None:
        return None
    return str(val)

# Define the FastAPI router
def serialize_value(val):
    """Main serialization function for all types"""
    if pd.isna(val):
        return None
    # Handle pandas Period objects
    elif isinstance(val, pd.Period):
        return val.asfreq('D').strftime('%Y-%m-%d')
    # Handle both datetime and Timestamp objects
    elif isinstance(val, (pd.Timestamp, datetime)):
        # Convert to ISO format and truncate microseconds if present
        iso_date = val.isoformat()
        if '.' in iso_date:
            iso_date = iso_date.split('.')[0] + 'Z'
        return iso_date
    elif isinstance(val, str):
        # Try to parse string as datetime if it looks like a date
        try:
            if 'T' in val or '-' in val:
                parsed_date = pd.to_datetime(val)
                return parsed_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        except:
            return val
    elif isinstance(val, np.integer): # Handle numpy integer types
        return int(val)
    elif isinstance(val, np.floating): # Handle numpy float types
        return float(val)
    elif isinstance(val, np.bool_): # Handle numpy boolean types
        return bool(val)
    elif isinstance(val, (np.ndarray, list)): # Handle numpy arrays and lists
        return [serialize_value(v) for v in val]
    elif isinstance(val, dict): # Handle dictionaries
        return {k: serialize_value(v) for k, v in val.items()}
    elif isinstance(val, pd.Index): # Handle pandas Index objects
        return serialize_value(val.tolist())
    elif isinstance(val, pd.Series): # Handle pandas Series objects
        return serialize_value(val.tolist())
    elif isinstance(val, pd.DataFrame): # Handle pandas DataFrame objects
        return serialize_value(val.to_dict('records'))
    elif isinstance(val, pd.DateOffset): # Handle pandas DateOffset objects
        return str(val)
    elif isinstance(val, pd.Interval): # Handle pandas Interval objects
        return str(val)
    return serialize_neo4j_value(val) # Fallback to Neo4j serialization


# Define the FastAPI router
@router.post("/start_session")
async def start_session():
    session_id = str(uuid.uuid4()) # Generate a unique session ID
    chat_histories[session_id] = {
        "questions": [],
        "answers": []
    }
    return JSONResponse(content={"session_id": session_id}) # Return the session ID as JSON


@router.post("/chat")
async def chat(question: Question):
    try:
        # Extract the required information from the question object
        session_id = question.session_id # Extract the session ID
        user_question = question.question # Extract the user question
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Extract the current timestamp

        # Check if the session ID is valid
        if session_id not in chat_histories:
            return JSONResponse(
                content={"error": "Invalid session ID"},
                status_code=400
            )

        # Add the user question to the chat history
        chat_histories[session_id]["questions"].append(user_question)

        # Get the query generation response
        query_response = generate_query(user_question, session_id, question)

        chat_histories[session_id]["answers"].append(query_response)

        try:
            # Return the response as a JSON object
            return JSONResponse(content={
                "session_id": session_id,
                "result": query_response
            })
        # Handle any exceptions
        except Exception as e:
            return JSONResponse(
                content={
                    "error": f"Error processing response: {str(e)}",
                    "raw_response": query_response
                },
                status_code=500
            )
    # Handle any exceptions
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            content={
                "error": "An error occurred processing your request",
                "detail": str(e)
            },
            status_code=500
        )
