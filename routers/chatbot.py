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

# Create FastAPI instance
router = APIRouter()

# Load environment variables
load_dotenv()

# Get environment variables
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

# In-memory storage for chat histories
chat_histories: Dict[str, Dict[str, List[str]]] = {}

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


# Define the FastAPI router
@router.post("/start_session")
async def start_session():
    session_id = str(uuid.uuid4()) # Generate a unique session ID
    chat_histories[session_id] = {
        "questions": [],
        "answers": []
    }
    return JSONResponse(content={"session_id": session_id}) # Return the session ID as JSON
