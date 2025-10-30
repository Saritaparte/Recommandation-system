"""
FastAPI Real-time Recommendation System API
With ranking logic, caching, and monitoring
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import pickle
import json
from datetime import datetime
from collections import defaultdict
import numpy as np

# ==================== REQUEST/RESPONSE MODELS ====================
class UserProfile(BaseModel):
    user_id: int
    preferences: Optional[Dict[str, float]] = {}
    history: Optional[List[int]] = []

class RecommendationRequest(BaseModel):
    user_id: int
    n_items: int = Field(default=10, ge=1, le=100)
    strategy: str = Field(default="hybrid", pattern="^(popularity|collaborative|svd|hybrid)$")
    exclude_items: Optional[List[int]] = []
    boost_categories: Optional[Dict[str, float]] = {}

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict]
    strategy_used: str
    timestamp: str
    processing_time_ms: float

class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    interaction_type: str
    value: Optional[float] = None

# ==================== IN-MEMORY CACHE ====================
class RecommendationCache:
    """Simple in-memory cache for recommendations"""
    
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = ttl_seconds
        
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        self.cache.clear()

# ==================== RANKING & RERANKING ====================
class RecommendationRanker:
    """Apply business rules and re-ranking logic"""
    
    @staticmethod
    def apply_business_rules(recommendations: List[int], 
                            item_metadata: Dict,
                            boost_categories: Dict[str, float] = None,
                            diversity_weight: float = 0.2) -> List[int]:
        """Apply business rules"""
        if not boost_categories:
            return recommendations
        
        scored_items = []
        for item_id in recommendations:
            score = 1.0
            if item_id in item_metadata:
                category = item_metadata[item_id].get('category')
                if category in boost_categories:
                    score *= boost_categories[category]
            scored_items.append((item_id, score))
        
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in scored_items]
    
    @staticmethod
    def ensure_diversity(recommendations: List[int],
                        item_metadata: Dict,
                        max_per_category: int = 3) -> List[int]:
        """Ensure category diversity"""
        category_counts = defaultdict(int)
        diverse_recs = []
        
        for item_id in recommendations:
            if item_id in item_metadata:
                category = item_metadata[item_id].get('category', 'unknown')
                if category_counts[category] < max_per_category:
                    diverse_recs.append(item_id)
                    category_counts[category] += 1
            else:
                diverse_recs.append(item_id)
                
        return diverse_recs

# ==================== FASTAPI APPLICATION ====================
app = FastAPI(
    title="Recommendation System API",
    description="Real-time recommendations with multiple strategies",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
cache = RecommendationCache(ttl_seconds=300)
ranker = RecommendationRanker()

models = {
    'popularity': None,
    'collaborative': None,
    'svd': None,
    'hybrid': None
}

item_metadata = {}
user_interactions = defaultdict(list)

# ==================== STARTUP & SHUTDOWN ====================
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("Loading recommendation models...")
    
    global models, item_metadata
    
    try:
        with open('models/popularity.pkl', 'rb') as f:
            models['popularity'] = pickle.load(f)
        with open('models/collaborative.pkl', 'rb') as f:
            models['collaborative'] = pickle.load(f)
        with open('models/svd.pkl', 'rb') as f:
            models['svd'] = pickle.load(f)
        with open('models/hybrid.pkl', 'rb') as f:
            models['hybrid'] = pickle.load(f)
        
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load models: {e}")
        print("API will run in demo mode")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down recommendation API...")
    cache.clear()

# ==================== HEALTH CHECK ====================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Recommendation System API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {k: v is not None for k, v in models.items()}
    }

# ==================== STRATEGY IMPLEMENTATIONS ====================
def generate_popularity_recommendations(request: RecommendationRequest) -> List[int]:
    """Generate popularity-based recommendations"""
    if models['popularity']:
        return models['popularity'].recommend(
            n_items=request.n_items,
            exclude_items=request.exclude_items
        )
    
    # Demo fallback
    return list(range(1, request.n_items + 1))

def generate_collaborative_recommendations(request: RecommendationRequest) -> List[int]:
    """Generate collaborative filtering recommendations"""
    if models['collaborative']:
        return models['collaborative'].recommend_item_based(
            request.user_id,
            n_items=request.n_items
        )
    
    # Demo fallback
    np.random.seed(request.user_id)
    return np.random.randint(1, 200, request.n_items).tolist()

def generate_svd_recommendations(request: RecommendationRequest) -> List[int]:
    """Generate SVD-based recommendations"""
    if models['svd']:
        return models['svd'].recommend(
            request.user_id,
            n_items=request.n_items,
            exclude_items=request.exclude_items
        )
    
    # Demo fallback
    np.random.seed(request.user_id + 1)
    return np.random.randint(1, 200, request.n_items).tolist()

def generate_hybrid_recommendations(request: RecommendationRequest) -> List[int]:
    """Generate hybrid recommendations"""
    if models['hybrid']:
        return models['hybrid'].recommend(
            request.user_id,
            n_items=request.n_items,
            exclude_items=request.exclude_items
        )
    
    # Demo fallback: combine multiple strategies
    pop_recs = generate_popularity_recommendations(request)
    cf_recs = generate_collaborative_recommendations(request)
    
    # Simple hybrid: interleave results
    hybrid = []
    for i in range(request.n_items):
        if i < len(pop_recs) and pop_recs[i] not in hybrid:
            hybrid.append(pop_recs[i])
        if i < len(cf_recs) and cf_recs[i] not in hybrid:
            hybrid.append(cf_recs[i])
    
    return hybrid[:request.n_items]

def format_recommendations(item_ids: List[int]) -> List[Dict]:
    """Format recommendations with metadata"""
    formatted = []
    for rank, item_id in enumerate(item_ids, 1):
        item_data = {
            "item_id": item_id,
            "rank": rank,
            "score": 1.0 / rank
        }
        
        if item_id in item_metadata:
            item_data.update(item_metadata[item_id])
        
        formatted.append(item_data)
    
    return formatted

# ==================== MAIN RECOMMENDATION ENDPOINT ====================
@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for a user"""
    start_time = datetime.now()
    
    # Check cache
    cache_key = f"{request.user_id}_{request.strategy}_{request.n_items}"
    cached = cache.get(cache_key)
    if cached:
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=cached,
            strategy_used=f"{request.strategy} (cached)",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=0
        )
    
    # Generate recommendations
    try:
        if request.strategy == "popularity":
            recommendations = generate_popularity_recommendations(request)
        elif request.strategy == "collaborative":
            recommendations = generate_collaborative_recommendations(request)
        elif request.strategy == "svd":
            recommendations = generate_svd_recommendations(request)
        elif request.strategy == "hybrid":
            recommendations = generate_hybrid_recommendations(request)
        else:
            raise HTTPException(status_code=400, detail="Invalid strategy")
        
        # Apply business rules
        if request.boost_categories:
            recommendations = ranker.apply_business_rules(
                recommendations,
                item_metadata,
                request.boost_categories
            )
        
        # Ensure diversity
        recommendations = ranker.ensure_diversity(recommendations, item_metadata)
        
        # Format response
        formatted_recs = format_recommendations(recommendations)
        
        # Cache results
        cache.set(cache_key, formatted_recs)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=formatted_recs,
            strategy_used=request.strategy,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

# ==================== USER FEEDBACK ENDPOINT ====================
@app.post("/feedback")
async def record_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """Record user feedback"""
    
    user_interactions[feedback.user_id].append({
        "item_id": feedback.item_id,
        "interaction_type": feedback.interaction_type,
        "value": feedback.value,
        "timestamp": datetime.now().isoformat()
    })
    
    # Invalidate cache
    cache_key_pattern = f"{feedback.user_id}_"
    keys_to_remove = [k for k in cache.cache.keys() if k.startswith(cache_key_pattern)]
    for key in keys_to_remove:
        if key in cache.cache:
            del cache.cache[key]
    
    return {
        "status": "success",
        "message": "Feedback recorded",
        "user_id": feedback.user_id
    }

# ==================== BATCH RECOMMENDATIONS ====================
@app.post("/recommend/batch")
async def get_batch_recommendations(user_ids: List[int], n_items: int = 10):
    """Get recommendations for multiple users"""
    
    results = {}
    for user_id in user_ids:
        try:
            request = RecommendationRequest(user_id=user_id, n_items=n_items)
            recommendations = generate_hybrid_recommendations(request)
            results[user_id] = format_recommendations(recommendations)
        except Exception as e:
            results[user_id] = {"error": str(e)}
    
    return {
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

# ==================== SIMILAR ITEMS ENDPOINT ====================
@app.get("/items/{item_id}/similar")
async def get_similar_items(item_id: int, n_items: int = Query(default=10, ge=1, le=50)):
    """Get similar items"""
    
    if models['collaborative'] and hasattr(models['collaborative'], 'item_similarity'):
        try:
            similar = models['collaborative'].item_similarity.loc[item_id].sort_values(ascending=False)[1:n_items+1]
            return {
                "item_id": item_id,
                "similar_items": [
                    {"item_id": int(idx), "similarity": float(score)} 
                    for idx, score in similar.items()
                ]
            }
        except:
            pass
    
    # Fallback
    np.random.seed(item_id)
    similar_ids = np.random.randint(1, 200, n_items).tolist()
    return {
        "item_id": item_id,
        "similar_items": [
            {"item_id": sid, "similarity": float(np.random.random())} 
            for sid in similar_ids
        ]
    }

# ==================== TRENDING ITEMS ====================
@app.get("/trending")
async def get_trending_items(timeframe: str = Query(default="24h", pattern="^(1h|24h|7d|30d)$"),
                            n_items: int = Query(default=10, ge=1, le=100)):
    """Get trending items"""
    
    item_counts = defaultdict(int)
    for user_id, interactions in user_interactions.items():
        for interaction in interactions[-100:]:
            item_counts[interaction['item_id']] += 1
    
    trending = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:n_items]
    
    return {
        "timeframe": timeframe,
        "trending_items": [
            {"item_id": item_id, "interaction_count": count}
            for item_id, count in trending
        ]
    }

# ==================== USER PROFILE ====================
@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Get user profile"""
    
    interactions = user_interactions.get(user_id, [])
    
    return {
        "user_id": user_id,
        "total_interactions": len(interactions),
        "recent_interactions": interactions[-10:],
        "top_categories": {}
    }

# ==================== STATISTICS ====================
@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    
    total_users = len(user_interactions)
    total_interactions = sum(len(interactions) for interactions in user_interactions.values())
    cache_size = len(cache.cache)
    
    return {
        "total_users": total_users,
        "total_interactions": total_interactions,
        "cache_size": cache_size,
        "models_status": {k: "loaded" if v else "not_loaded" for k, v in models.items()},
        "uptime": "running"
    }

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    print("🚀 Starting Recommendation System API...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("❤️ Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )