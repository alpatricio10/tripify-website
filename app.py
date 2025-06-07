# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
from backend.rag import ItineraryRAG
from backend.enhanced_itinerary import EnhancedItineraryGenerator
from backend.database import db
from datetime import datetime
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)
CORS(app)

# Initialize RAG system
logger.info("Initializing RAG system...")
try:
    rag_system = ItineraryRAG(
        gemini_api_key=os.getenv('GEMINI_API_KEY'),
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        pinecone_environment=os.getenv('PINECONE_ENVIRONMENT'),
        mongo_uri=os.getenv('MONGO_URI')
    )
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {str(e)}", exc_info=True)
    raise

# Initialize Enhanced Itinerary Generator
logger.info("Initializing Enhanced Itinerary Generator...")
try:
    enhanced_generator = EnhancedItineraryGenerator(
        rag_system=rag_system,
        graph_api_url=None  # We'll use local Neo4j directly
    )
    logger.info("Enhanced Itinerary Generator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Enhanced Itinerary Generator: {str(e)}", exc_info=True)
    raise

# Load existing itineraries
logger.info("Loading existing itineraries...")
try:
    rag_system.load_llm_responses()
    logger.info("Existing itineraries loaded successfully")
except Exception as e:
    logger.error(f"Failed to load existing itineraries: {str(e)}", exc_info=True)
    raise

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/booking')
def booking():
    return render_template('booking.html')

@app.route('/trip')
def trip():
    return render_template('Trip.html')

@app.route('/itineraries')
def itineraries():
    return render_template('itineraries.html')

@app.route('/api/v1/cities/popular', methods=['GET'])
def get_popular_cities():
    """Get list of all available cities"""
    try:
        with db.get_session() as session:
            query = """
            MATCH (c:City)
            OPTIONAL MATCH (p:Post)-[:IN]->(c)
            WITH c, count(DISTINCT p) as post_count
            RETURN c,
                   post_count as popularity
            ORDER BY post_count DESC
            """
            result = session.run(query)
            cities = []
            for record in result:
                city = dict(record["c"])
                city["popularity"] = record["popularity"]
                cities.append(city)
            return jsonify(cities)
    except Exception as e:
        logger.error(f"Error fetching cities: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/itineraries/recommended', methods=['GET'])
def get_recommended_itineraries():
    """Get personalized itinerary recommendations based on user preferences"""
    try:
        user_id = request.args.get('user_id')
        
        with db.get_session() as session:
            # If no user_id provided, use a default user
            if not user_id:
                # First check if default user exists
                default_user_query = """
                MATCH (u:User {id: 'default_user'})
                RETURN u
                """
                default_user = session.run(default_user_query).single()
                
                if not default_user:
                    # Create default user if doesn't exist
                    create_default_user_query = """
                    CREATE (u:User {
                        id: 'default_user',
                        name: 'Default User',
                        username: 'default_user',
                        join_date: datetime()
                    })
                    RETURN u
                    """
                    session.run(create_default_user_query)
                
                user_id = 'default_user'

            # Get user's preferred cities from their preferences
            preferences_query = """
            MATCH (u:User {id: $user_id})-[:FAVORITE]->(c:City)
            RETURN collect(c.id) as preferred_cities
            """
            preferences_result = session.run(preferences_query, user_id=user_id)
            preferences_record = preferences_result.single()
            preferred_cities = preferences_record["preferred_cities"] if preferences_record else []

            if not preferred_cities:
                return jsonify([])

            query = """
            MATCH (p:Post)-[:IN]->(city:City)
            WHERE city.id IN $preferred_cities
            
            MATCH (p)-[:INCLUDES]->(a:Activity)
            WITH p, city, collect(a) as activities
            
            OPTIONAL MATCH (u:User {id: $user_id})-[l:LIKED]->(p)
            WITH p, city, activities, l IS NOT NULL as is_liked
            
            RETURN p, city, activities, is_liked
            ORDER BY p.created_at DESC
            LIMIT 10
            """
            
            result = session.run(
                query,
                preferred_cities=preferred_cities,
                user_id=user_id
            )
            
            itineraries = []
            for record in result:
                itinerary = {
                    "post": dict(record["p"]),
                    "city": dict(record["city"]),
                    "activities": [dict(activity) for activity in record["activities"]],
                    "is_liked": record["is_liked"]
                }
                itineraries.append(itinerary)
            
            return jsonify(itineraries)
    except Exception as e:
        logger.error(f"Error fetching recommended itineraries: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/users/current/preferences', methods=['GET'])
def get_current_user_preferences():
    """Get preferences for the current user"""
    try:
        with db.get_session() as session:
            # First check if default user exists
            check_user_query = """
            MATCH (u:User {id: 'default_user'})
            RETURN u
            """
            default_user = session.run(check_user_query).single()
            
            if not default_user:
                # Create default user if doesn't exist
                create_user_query = """
                CREATE (u:User {
                    id: 'default_user',
                    name: 'Default User',
                    username: 'default_user',
                    join_date: datetime()
                })
                RETURN u
                """
                session.run(create_user_query)
            
            # Get user's favorite cities
            query = """
            MATCH (u:User {id: 'default_user'})
            OPTIONAL MATCH (u)-[:FAVORITE]->(c:City)
            RETURN collect(c.id) as favorite_cities
            """
            result = session.run(query)
            record = result.single()
            
            if record:
                preferences = {
                    "favorite_cities": record["favorite_cities"] or [],
                    "travel_style": "cultural"  # Default value
                }
                return jsonify(preferences)
            return jsonify({"favorite_cities": [], "travel_style": "cultural"})
    except Exception as e:
        logger.error(f"Error fetching user preferences: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/users/current/preferences', methods=['POST'])
def update_current_user_preferences():
    """Update preferences for the current user"""
    try:
        preferences = request.json
        with db.get_session() as session:
            # First check if default user exists
            check_user_query = """
            MATCH (u:User {id: 'default_user'})
            RETURN u
            """
            default_user = session.run(check_user_query).single()
            
            if not default_user:
                # Create default user if doesn't exist
                create_user_query = """
                CREATE (u:User {
                    id: 'default_user',
                    name: 'Default User',
                    username: 'default_user',
                    join_date: datetime()
                })
                RETURN u
                """
                session.run(create_user_query)

            # First, remove all existing FAVORITE relationships
            delete_query = """
            MATCH (u:User {id: 'default_user'})-[f:FAVORITE]->(c:City)
            DELETE f
            """
            session.run(delete_query)

            # Then create new FAVORITE relationships for selected cities
            if preferences.get('favorite_cities'):
                create_query = """
                MATCH (u:User {id: 'default_user'})
                WITH u
                UNWIND $favorite_cities as city_id
                MATCH (c:City {id: city_id})
                MERGE (u)-[:FAVORITE {created_at: datetime()}]->(c)
                """
                session.run(create_query, favorite_cities=preferences.get('favorite_cities', []))

            # Get updated preferences
            get_query = """
            MATCH (u:User {id: 'default_user'})
            OPTIONAL MATCH (u)-[:FAVORITE]->(c:City)
            RETURN collect(c.id) as favorite_cities
            """
            result = session.run(get_query)
            record = result.single()
            
            if record:
                updated_preferences = {
                    "favorite_cities": record["favorite_cities"] or [],
                    "budget": preferences.get('budget', 2000)  # Default budget if not provided
                }
                return jsonify(updated_preferences)
            return jsonify({"favorite_cities": [], "budget": 2000})
    except Exception as e:
        logger.error(f"Error updating user preferences: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/generate-itinerary', methods=['POST'])
async def generate_itinerary():
    try:
        data = request.json
        logger.info(f"Received request data: {data}")
        
        destinations = data.get('destinations', [])
        dates = data.get('dates', {})
        preferences = data.get('preferences', {})
        budget = data.get('budget', '500')
        user_id = data.get('user_id')

        if not destinations:
            logger.error("No destinations provided in request")
            return jsonify({"error": "No destinations provided"}), 400

        # Get the main destination (last city in the list)
        main_destination = destinations[-1]
        
        # Extract categories from preferences
        categories = []
        if preferences.get('interests'):
            categories.extend(preferences['interests'])
        
        # Calculate number of days from dates
        days = 1
        if dates and 'start' in dates and 'end' in dates:
            start_date = datetime.strptime(dates['start'], '%Y-%m-%d')
            end_date = datetime.strptime(dates['end'], '%Y-%m-%d')
            days = (end_date - start_date).days + 1

        # Generate enhanced itinerary
        itinerary = await enhanced_generator.generate_enhanced_itinerary(
            city=main_destination,
            categories=categories,
            days=days,
            user_id=user_id
        )

        return jsonify(itinerary)
    except Exception as e:
        logger.error(f"Error generating itinerary: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/itineraries/<itinerary_id>/like', methods=['POST'])
def like_itinerary(itinerary_id):
    """Like or unlike an itinerary"""
    try:
        with db.get_session() as session:
            # First check if default user exists
            check_user_query = """
            MATCH (u:User {id: 'default_user'})
            RETURN u
            """
            default_user = session.run(check_user_query).single()
            
            if not default_user:
                # Create default user if doesn't exist
                create_user_query = """
                CREATE (u:User {
                    id: 'default_user',
                    name: 'Default User',
                    username: 'default_user',
                    join_date: datetime()
                })
                RETURN u
                """
                session.run(create_user_query)

            # Check if the like relationship exists
            check_like_query = """
            MATCH (u:User {id: 'default_user'})-[l:LIKED]->(p:Post {id: $post_id})
            RETURN l
            """
            like_exists = session.run(check_like_query, post_id=itinerary_id).single()

            if like_exists:
                # Unlike: Remove the relationship and decrease vote count
                unlike_query = """
                MATCH (u:User {id: 'default_user'})-[l:LIKED]->(p:Post {id: $post_id})
                DELETE l
                WITH p
                SET p.votes = p.votes - 1
                RETURN p
                """
                result = session.run(unlike_query, post_id=itinerary_id).single()
                liked = False
            else:
                # Like: Create the relationship and increase vote count
                like_query = """
                MATCH (u:User {id: 'default_user'})
                MATCH (p:Post {id: $post_id})
                MERGE (u)-[l:LIKED {created_at: datetime()}]->(p)
                WITH p
                SET p.votes = COALESCE(p.votes, 0) + 1
                RETURN p
                """
                result = session.run(like_query, post_id=itinerary_id).single()
                liked = True

            if result:
                return jsonify({
                    "votes": result["p"].get("votes", 0),
                    "liked": liked
                })
            return jsonify({"error": "Itinerary not found"}), 404

    except Exception as e:
        logger.error(f"Error liking/unliking itinerary: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)