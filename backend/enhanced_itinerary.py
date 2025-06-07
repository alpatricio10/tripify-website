# backend/enhanced_itinerary.py
from typing import List, Dict, Optional
from datetime import datetime
import json
import logging
from backend.database import db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedItineraryGenerator:
    def __init__(self, 
                 rag_system,
                 graph_api_url: str = None):
        """
        Initialize the enhanced itinerary generator that combines RAG and graph-based recommendations
        
        Args:
            rag_system: Instance of ItineraryRAG system
            graph_api_url: Not used anymore, kept for backward compatibility
        """
        self.rag_system = rag_system
        self.logger = logging.getLogger(__name__)

    async def generate_enhanced_itinerary(self, 
                                        city: str, 
                                        categories: List[str], 
                                        days: int = 3,
                                        user_id: Optional[str] = None) -> Dict:
        """
        Generate an enhanced itinerary combining RAG and graph-based recommendations
        
        Args:
            city: Target city
            categories: List of categories to include
            days: Number of days for the itinerary
            user_id: Optional user ID for personalized recommendations
            
        Returns:
            Enhanced itinerary in JSON format
        """
        try:
            # 1. Get base itinerary from RAG system
            base_itinerary = self.rag_system.generate_structured_itinerary(
                city=city,
                categories=categories,
                days=days
            )

            # 2. Get popular activities from Neo4j
            popular_activities = self._get_popular_activities(city)

            # 3. Get personalized recommendations if user_id is provided
            personalized_recommendations = []
            if user_id:
                personalized_recommendations = self._get_personalized_recommendations(
                    user_id=user_id,
                    city=city
                )

            # 4. Enhance the itinerary with graph-based recommendations
            enhanced_itinerary = self._enhance_itinerary(
                base_itinerary=base_itinerary,
                popular_activities=popular_activities,
                personalized_recommendations=personalized_recommendations,
                days=days
            )

            return enhanced_itinerary

        except Exception as e:
            self.logger.error(f"Error generating enhanced itinerary: {str(e)}")
            raise

    def _get_popular_activities(self, city: str) -> List[Dict]:
        """Get popular activities for a city from Neo4j"""
        try:
            with db.get_session() as session:
                query = """
                MATCH (a:Activity)-[:IN]->(c:City {id: $city})
                OPTIONAL MATCH (u:User)-[:LIKED]->(p:Post)-[:INCLUDES]->(a)
                WITH a, count(DISTINCT u) as likes_count
                RETURN a, likes_count
                ORDER BY likes_count DESC
                LIMIT 10
                """
                result = session.run(query, city=city)
                activities = []
                for record in result:
                    activity = dict(record["a"])
                    activity["likes_count"] = record["likes_count"]
                    activities.append(activity)
                return activities
        except Exception as e:
            self.logger.error(f"Error fetching popular activities: {str(e)}")
            return []

    def _get_personalized_recommendations(self, user_id: str, city: str) -> Dict:
        """Get personalized recommendations for a user from Neo4j"""
        try:
            with db.get_session() as session:
                # Get recommended posts
                posts_query = """
                MATCH (u:User {id: $user_id})
                OPTIONAL MATCH (u)-[:LIKED]->(p:Post)
                WITH u, collect(p) as liked_posts
                MATCH (p:Post)-[:IN]->(c:City {id: $city})
                WHERE NOT p IN liked_posts
                RETURN p
                ORDER BY p.created_at DESC
                LIMIT 5
                """
                posts_result = session.run(posts_query, user_id=user_id, city=city)
                recommended_posts = [dict(record["p"]) for record in posts_result]

                # Get user's liked activities
                activities_query = """
                MATCH (u:User {id: $user_id})-[:LIKED]->(p:Post)-[:INCLUDES]->(a:Activity)-[:IN]->(c:City {id: $city})
                RETURN a
                """
                activities_result = session.run(activities_query, user_id=user_id, city=city)
                liked_activities = [dict(record["a"]) for record in activities_result]

                return {
                    "recommended_posts": recommended_posts,
                    "liked_activities": liked_activities
                }
        except Exception as e:
            self.logger.error(f"Error fetching personalized recommendations: {str(e)}")
            return []

    def _enhance_itinerary(self, 
                          base_itinerary: Dict,
                          popular_activities: List[Dict],
                          personalized_recommendations: Dict,
                          days: int) -> Dict:
        """Enhance the base itinerary with graph-based recommendations"""
        enhanced_itinerary = base_itinerary.copy()

        # Add popular activities as alternatives
        enhanced_itinerary["popular_activities"] = popular_activities

        # Add personalized recommendations if available
        if personalized_recommendations:
            enhanced_itinerary["personalized_recommendations"] = personalized_recommendations

        # Add social proof
        enhanced_itinerary["social_proof"] = {
            "total_likes": sum(activity.get("likes_count", 0) for activity in popular_activities),
            "user_reviews": len(popular_activities)
        }

        # Add additional tips based on popular activities
        enhanced_itinerary["additional_tips"] = self._generate_additional_tips(
            popular_activities=popular_activities,
            personalized_recommendations=personalized_recommendations
        )

        return enhanced_itinerary

    def _generate_additional_tips(self, 
                                popular_activities: List[Dict],
                                personalized_recommendations: Dict) -> List[str]:
        """Generate additional tips based on popular activities and recommendations"""
        tips = []

        # Add tips based on popular activities
        for activity in popular_activities:
            if activity.get("likes_count", 0) > 10:
                tips.append(f"Popular choice: {activity['name']} is highly recommended by other travelers")

        # Add tips based on personalized recommendations
        if personalized_recommendations:
            liked_activities = personalized_recommendations.get("liked_activities", [])
            if liked_activities:
                tips.append("Based on your preferences, you might enjoy similar activities to your previous trips")

        return tips