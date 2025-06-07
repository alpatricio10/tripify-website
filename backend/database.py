# backend/database.py
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Neo4jDatabase:
    def __init__(self):
        # Neo4j Aura cloud instance
        self.uri = os.getenv("NEO4J_URI", "neo4j+s://cdfd2b56.databases.neo4j.io")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "JTV8QRSY4LrutGxPze-obf0FQg-3lbCNetYtjZ3bkXs")
        self.driver = None
        logger.info(f"Initializing Neo4j connection to {self.uri}")
        
        # Try to connect immediately
        self.connect()

    def connect(self):
        """Connect to Neo4j database"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Neo4j (attempt {attempt + 1}/{max_retries})...")
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password),
                    max_connection_lifetime=30,
                    max_connection_pool_size=50,
                    connection_timeout=20
                )
                
                # Verify connection with a simple query
                with self.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    if test_value == 1:
                        logger.info("Successfully connected to Neo4j!")
                        return self.driver
                    else:
                        logger.error("Connection test failed - unexpected result")
                        raise Exception("Connection test failed")
                        
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Could not connect to Neo4j.")
                    raise

    def close(self):
        """Close the database connection"""
        if self.driver:
            logger.info("Closing Neo4j connection")
            self.driver.close()

    def get_session(self):
        """Get a new database session"""
        if not self.driver:
            self.connect()
        return self.driver.session()

# Create a global database instance
db = Neo4jDatabase() 