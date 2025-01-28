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
from routers import prompts
from routers.prompts import db_structure
from fastapi.staticfiles import StaticFiles

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
db_structure_prompt = db_structure

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

# Get the current file's directory and set up static path
CURRENT_DIR = Path(__file__).resolve().parent
STATIC_DIR = CURRENT_DIR.parent / "static" / "temp"
os.makedirs(STATIC_DIR, exist_ok=True)

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
        "query": "OPTIONAL MATCH query here"
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
    QUERY_CHAT_PROMPT = """Role and Context:
                        You are TravelGuru, an AI assistant that helps users with travel-related queries about Sri Lanka.
                        
                        Recent Conversation History: (This contains the recent questions user asked)
                        -------------------------
                        {history_text}
                        
                         Database Context:
                        ---------------
                        Current Graph Database Structure: {graph_db_structure}
                        
                        Reference Examples (Not Part of Current Conversation or history):
                        -------------------------
                        Reference Examples : {similar_chunks}
                        
                          
                       ⚠️ Note: These examples are ONLY for cypher query generation reference. Do not use Reference Examples as conversation history or general content.
                       Very Very Important: ⚠️⚠️ Do Not ever use the same reference examples to generate your answer, because it is not the answer user is expecting. 
                       it is given to get an idea of the cypher query structure.
                        Query Requirements:
                        -----------------
                        1. All queries must start with OPTIONAL MATCH
                        2. Only use nodes, elements, and relationships defined in the provided graph structure
                        3. Support cross-table filtering when necessary
                        
                        Today Date:
                        ---------------
                        {date}
                        
                        Today Time:
                        ______________
                        {time}
                        
                        Input Question:
                        --------------
                        {question}
                        
                        Response Format:
                        --------------
                        IMPORTANT: Return ONLY the following structure with no prefix text like 'Template:' or 'JSON:':
                         {template_cypher}
                
                        ⚠️ DO NOT include any text before or after the response structure
                        ⚠️ DO NOT add labels like 'Template:', 'Response:', or 'JSON:'
                        ⚠️ DO NOT include any additional formatting or explanation
                        ⚠️ ONLY return the exact structure shown above
                        
                        Response Guidelines:
                        ------------------
                        1. text_explanation:
                           - Provide natural language explanations
                           - Can include emojis for better readability
                           - Must NOT contain code or queries
                           - For general questions, provide complete, knowledgeable answers here
                           - For data-related questions, focus on insights and business impact
                           - Never mention querying or accessing a database
                           - Maintain a conversational, helpful tone
                           - Provide natural language explanations ONLY about Sysco Labs
                           - Must be strictly related to Sysco Labs operations and services
                           - Must NOT contain any political, controversial, or non-Sysco content
                           - For general questions, ONLY provide answers related to Sysco Labs
                           - Maintain a professional, business-focused tone
                        
                        2. query_generation_status:
                           - 'Yes': If a Cypher query can be generated
                           - 'No': If question cannot be answered with a query
                           - If 'No', query field should be empty
                        """

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

        # Format the prompt with all required values
    formatted_prompt = QUERY_CHAT_PROMPT.format(
        date=question.date,
        time=question.time,
        history_text=history_text,
        graph_db_structure=db_structure_prompt,
        similar_chunks=similar_text,
        question=user_question,
        template_cypher=TEMPLATE_CYPHER,
        settings_prompt=settings_prompt
    )


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
                            "text": formatted_prompt  # Add the formatted prompt to the content
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
def execute_cypher_query(driver, query):
    """Execute a Cypher query and return serialized results"""
    try:
        with driver.session() as session:
            result = session.run(query)
            # Convert Neo4j records to dictionaries and serialize the values
            records = []
            for record in result:
                # Convert each record to a dict and serialize all values
                record_dict = {}
                for key, value in dict(record).items(): # Convert to dict to access items
                    record_dict[key] = serialize_value(value)
                records.append(record_dict) # Append the serialized record to the list
            return records
    except Exception as e: # Handle any exceptions
        print(f"Error executing query: {str(e)}")
        return None

# Function to generate a Cypher query based on the user question
def convert_to_html_table(data):
    if not data:
        return ""

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Function to convert email to clickable link
    def make_email_link(value):
        if pd.isna(value):
            return ""
        value_str = str(value).strip()
        # Check if the value looks like an email address
        if '@' in value_str and '.' in value_str.split('@')[1]:
            return f'<a href="mailto:{value_str}" style="color: #0066cc; text-decoration: none;">{value_str}</a>'
        return value_str

    # Identify email columns (any column with 'email' in the name, case-insensitive)
    email_columns = [col for col in df.columns if 'email' in col.lower()]

    # Apply the email link conversion to identified columns
    for col in email_columns:
        df[col] = df[col].apply(make_email_link)

    # Generate HTML table
    html_table = df.to_html(
        classes=['table', 'table-striped', 'table-hover', 'table-bordered'],
        index=False,
        escape=False,
        table_id='result-table'
    )

    # Add CSS to make column separators visible and apply additional styling
    html_table = html_table.replace('<table',
        '<table style="border-collapse: collapse; border: 1px solid black; width: 100%;"')
    html_table = html_table.replace('<th',
        '<th style="border: 1px solid black; text-align: center; padding: 8px;"')
    html_table = html_table.replace('<td',
        '<td style="border: 1px solid black; text-align: right; padding: 8px;"')

    return html_table

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

        try:
            try:
                # Parse the response as JSON
                response_dict = json.loads(query_response)
            except json.JSONDecodeError:
                try:
                    # Try to parse the response as a Python dict
                    response_dict = ast.literal_eval(query_response)
                except (ValueError, SyntaxError):
                    raise ValueError(
                        "Could not parse response as JSON or Python dict")  # Raise an error if parsing fails

            # Validate response format
            required_fields = [
                "text_explanation",
                "query_generation_status",
                "query"
            ]

            # Check if the response dictionary contains all required fields
            if not isinstance(response_dict, dict) or not all(
                    key in response_dict for key in required_fields):
                raise ValueError("Invalid response format")

            # Initialize the result
            result = {
                "text_explanation": response_dict["text_explanation"],
                "query_generation_status": response_dict["query_generation_status"],
                "query": response_dict["query"],
                "data": None,
                "html_table_data": ""
            }

            # Execute the query if status is "Yes"
            if response_dict["query_generation_status"].lower() == "yes" and response_dict["query"]:
                query_result = execute_cypher_query(driver, response_dict["query"])
                # Ensure the entire result is JSON serializable
                result["data"] = json.loads(json.dumps(query_result, default=str))

                # Generate HTML table from the data
                result["html_table_data"] = convert_to_html_table(query_result)

                # Save HTML table if it exists
                if result["html_table_data"]:
                    try:
                        table_filename = f'table_{session_id}_{timestamp}.html'  # Generate a unique filename
                        file_path = STATIC_DIR / table_filename  # Set the file path

                        # Write the file
                        with open(file_path, 'w', encoding='utf-8') as f:  # Open the file in write mode
                            f.write(result["html_table_data"])  # Write the HTML table data to the file

                        # Set the correct URL using the base URL from environment
                        base_url = os.getenv('API_BASE_URL', 'https://codefest-backend.azurewebsites.net')
                        result["table_file_url"] = f"{base_url}/static/temp/{table_filename}"

                    except Exception as e:  # Handle any exceptions
                        print(f"Error saving table file: {str(e)}")

                chat_histories[session_id]["answers"].append(query_response)


            # Return the response as a JSON object
            return JSONResponse(content={
                "session_id": session_id,
                "result": result
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
