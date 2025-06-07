import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import json
import os
import google.generativeai as genai
import logging
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import json_util

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ItineraryRAG:
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 gemini_api_key: str = None,
                 pinecone_api_key: str = None,
                 pinecone_environment: str = None,
                 index_name: str = "itinerary-events",
                 mongo_uri: str = None,
                 mongo_db: str = "itinerary_db",
                 mongo_collection: str = "itineraries"):
        """
        Initialize the RAG system for itinerary generation
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
            gemini_api_key: API key for Gemini
            pinecone_api_key: API key for Pinecone
            pinecone_environment: Pinecone environment
            index_name: Name of the Pinecone index
            mongo_uri: MongoDB connection URI
            mongo_db: MongoDB database name
            mongo_collection: MongoDB collection name
        """
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.llm_responses = {}
        
        # Initialize Gemini
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize Pinecone
        if pinecone_api_key and pinecone_environment:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index_name = index_name
            self._initialize_pinecone_index()
        else:
            raise ValueError("Pinecone API key and environment are required")
            
        # Initialize MongoDB
        if mongo_uri:
            self.mongo_client = MongoClient(mongo_uri)
            self.mongo_db = self.mongo_client[mongo_db]
            self.mongo_collection = self.mongo_db[mongo_collection]
        else:
            raise ValueError("MongoDB URI is required")
        
    def _initialize_pinecone_index(self):
        """Initialize or connect to Pinecone index"""
        if self.index_name not in self.pc.list_indexes().names():
            # Create index if it doesn't exist
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.pinecone_index = self.pc.Index(self.index_name)
    
    def load_llm_responses(self):
        """
        Load existing LLM responses from MongoDB
        """
        try:
            # Query MongoDB for all itineraries
            itineraries = self.mongo_collection.find({})
            
            for itinerary in itineraries:
                city = itinerary.get('city')
                category = itinerary.get('type')
                
                if city and category:
                    if city not in self.llm_responses:
                        self.llm_responses[city] = {}
                    
                    # Store the entire itinerary document
                    self.llm_responses[city][category] = itinerary
                    logger.info(f"Loaded response for {city} - {category}")
            
            logger.info(f"Loaded {len(self.llm_responses)} cities with their itineraries")
            
        except Exception as e:
            logger.error(f"Error loading responses from MongoDB: {str(e)}")
            raise
    
    def retrieve_relevant_events(self, query: str, city: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant events based on the query using Pinecone
        
        Args:
            query: Search query
            city: Target city
            k: Number of results to return
            
        Returns:
            List of relevant events
        """
        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in Pinecone
        results = self.pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=k * 2,  # Get more results to filter by city
            include_metadata=True
        )
        
        # Filter results for the specified city
        relevant_events = []
        for match in results.matches:
            if match.metadata['city'] == city:
                # The metadata already contains all the event information
                relevant_events.append(match.metadata)
                if len(relevant_events) >= k:
                    break
        
        return relevant_events
    
    def generate_structured_itinerary(self, city: str, categories: List[str], days: int = 3) -> Dict:
        """
        Generate a structured itinerary combining multiple categories
        
        Args:
            city: Target city
            categories: List of categories to include (e.g., ["City Sight Seeing", "Beaches", "Historical Places"])
            days: Number of days for the itinerary
            
        Returns:
            Structured itinerary in JSON format
        """
        # Get relevant context from MongoDB for each category
        contexts = []
        for category in categories:
            if city in self.llm_responses and category in self.llm_responses[city]:
                # Convert MongoDB document to JSON-serializable format
                context = self.llm_responses[city][category]
                # Remove ObjectId and convert to dict
                context_dict = {
                    'city': context.get('city'),
                    'type': context.get('type'),
                    'days': context.get('days'),
                    'itinerary': context.get('itinerary')
                }
                contexts.append(context_dict)
        
        # Retrieve relevant events for each category
        all_relevant_events = []
        for category in categories:
            query = f"{category} activities in {city}"
            events = self.retrieve_relevant_events(query, city, k=5)  # Reduced k to avoid too many events
            all_relevant_events.extend(events)
        
        # Prepare prompt for Gemini
        prompt = f"""You are a travel itinerary expert who creates detailed and practical travel plans.

                    Generate a {days}-day itinerary for {city} combining the following categories: {', '.join(categories)}.

                    Previous successful itineraries for reference:
                    {json.dumps(contexts, indent=2)}

                    Available events:
                    {json.dumps(all_relevant_events, indent=2)}

                    Please create a detailed itinerary in the following JSON format:
                    {{
                        "itinerary": {{
                            "city": "{city}",
                            "categories": {json.dumps(categories)},
                            "days": {days},
                            "daily_plans": [
                                {{
                                    "day": 1,
                                    "activities": [
                                        {{
                                            "time": "HH:MM",
                                            "event_name": "string",
                                            "venue": "string",
                                            "description": "string",
                                            "duration": "string",
                                            "price": "string",
                                            "booking_info": "string",
                                            "category": "string"
                                        }}
                                    ]
                                }}
                            ],
                            "total_cost": "string",
                            "additional_tips": ["string"]
                        }}
                    }}

                    Make sure to:
                    - Include a mix of events from all categories
                    - Consider timing and location logistics
                    - Add variety to the activities
                    - Include practical information like prices and booking requirements
                    - Balance the activities across different categories
                    - Consider the time of day for each activity (e.g., beaches in the afternoon)"""
        
        # Generate itinerary using Gemini
        response = self.gemini_model.generate_content(prompt)
        
        try:
            # Clean the response text by removing markdown code block formatting
            response_text = response.text
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove trailing ```
            response_text = response_text.strip()
            
            # Parse the cleaned response as JSON
            itinerary_json = json.loads(response_text)
            
            # Store the generated itinerary in MongoDB
            document = {
                'city': city,
                'types': categories,  # Store all categories
                'days': days,
                'itinerary': itinerary_json
            }
            self.mongo_collection.insert_one(document)
            
            return itinerary_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {str(e)}")
            return {
                "error": "Failed to generate structured itinerary",
                "raw_response": response.text
            }

def main():
    try:
        # Get API keys from environment variables
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        
        pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
        
        mongo_uri = os.getenv('MONGODB_URI')
        
        
        if not all([gemini_api_key, pinecone_api_key, pinecone_environment, mongo_uri]):
            raise ValueError("Please set all required environment variables: GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, MONGODB_URI")
        
        # Initialize RAG system
        rag = ItineraryRAG(
            gemini_api_key=gemini_api_key,
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            mongo_uri=mongo_uri
        )
        
        # Load LLM responses from MongoDB
        rag.load_llm_responses()
        
        # Example usage with multiple categories
        categories = ["City Sight Seeing", "Beaches", "Historical Places"]
        itinerary = rag.generate_structured_itinerary("barcelona", categories, days=3)
        
        print(json.dumps(itinerary, indent=2))

        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
