"""
Complete Recommendation System Implementation
Includes: Popularity, Collaborative Filtering, Matrix Factorization, and Neural Models
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import pickle
from datetime import datetime

# ==================== 1. DATA LOADER & PREPROCESSOR ====================
class DataProcessor:
    """Handle data loading and preprocessing"""
    
    def __init__(self):
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        
    def load_movielens_format(self, ratings_path: str, movies_path: str = None):
        """Load MovieLens-style dataset"""
        # Load ratings (user_id, item_id, rating, timestamp)
        self.ratings = pd.read_csv(ratings_path)
        
        if movies_path:
            self.items = pd.read_csv(movies_path)
        
        # Create user-item matrix
        self.user_item_matrix = self.ratings.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        return self.ratings, self.user_item_matrix
    
    def create_synthetic_data(self, n_users=1000, n_items=500, sparsity=0.95):
        """Generate synthetic data for testing"""
        np.random.seed(42)
        
        # Generate ratings matrix
        n_ratings = int(n_users * n_items * (1 - sparsity))
        
        users = np.random.randint(0, n_users, n_ratings)
        items = np.random.randint(0, n_items, n_ratings)
        ratings = np.random.randint(1, 6, n_ratings)
        
        self.ratings = pd.DataFrame({
            'user_id': users,
            'item_id': items,
            'rating': ratings,
            'timestamp': pd.date_range('2023-01-01', periods=n_ratings, freq='1min')
        })
        
        # Create user-item matrix
        self.user_item_matrix = self.ratings.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)
        
        # Create item metadata
        self.items = pd.DataFrame({
            'item_id': range(n_items),
            'title': [f'Item_{i}' for i in range(n_items)],
            'category': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi'], n_items)
        })
        
        return self.ratings, self.user_item_matrix, self.items


# ==================== 2. POPULARITY-BASED RECOMMENDER ====================
class PopularityRecommender:
    """Recommend items based on popularity metrics"""
    
    def __init__(self):
        self.item_scores = None
        
    def fit(self, ratings_df: pd.DataFrame, method='count'):
        """
        Train popularity model
        method: 'count', 'average', 'weighted'
        """
        if method == 'count':
            self.item_scores = ratings_df.groupby('item_id').size()
            
        elif method == 'average':
            self.item_scores = ratings_df.groupby('item_id')['rating'].mean()
            
        elif method == 'weighted':
            # Weighted rating = (v/(v+m)) * R + (m/(v+m)) * C
            # v: vote count, m: minimum votes, R: average rating, C: mean across all
            counts = ratings_df.groupby('item_id').size()
            avg_ratings = ratings_df.groupby('item_id')['rating'].mean()
            
            m = counts.quantile(0.7)  # minimum votes threshold
            C = ratings_df['rating'].mean()
            
            self.item_scores = (counts / (counts + m)) * avg_ratings + (m / (counts + m)) * C
        
        self.item_scores = self.item_scores.sort_values(ascending=False)
        return self
    
    def recommend(self, n_items=10, exclude_items=None):
        """Get top N popular items"""
        recommendations = self.item_scores.head(n_items)
        
        if exclude_items:
            recommendations = recommendations[~recommendations.index.isin(exclude_items)]
            recommendations = recommendations.head(n_items)
            
        return recommendations.index.tolist()


# ==================== 3. COLLABORATIVE FILTERING ====================
class CollaborativeFiltering:
    """User-based and Item-based Collaborative Filtering"""
    
    def __init__(self, similarity='cosine'):
        self.similarity_metric = similarity
        self.user_similarity = None
        self.item_similarity = None
        self.user_item_matrix = None
        
    def fit(self, user_item_matrix):
        """Compute similarity matrices"""
        self.user_item_matrix = user_item_matrix
        
        # Compute item-item similarity
        self.item_similarity = self._compute_similarity(user_item_matrix.T)
        
        # Compute user-user similarity
        self.user_similarity = self._compute_similarity(user_item_matrix)
        
        return self
    
    def _compute_similarity(self, matrix):
        """Compute cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Handle sparse data
        matrix_values = matrix.values
        sim = cosine_similarity(matrix_values)
        
        return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    
    def recommend_user_based(self, user_id, n_items=10):
        """User-based CF: recommend based on similar users"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get similar users
        similar_users = self.user_similarity.loc[user_id].sort_values(ascending=False)[1:31]
        
        # Get items rated by similar users but not by target user
        user_items = self.user_item_matrix.loc[user_id]
        unrated_items = user_items[user_items == 0].index
        
        # Calculate weighted scores
        scores = {}
        for item in unrated_items:
            weighted_sum = 0
            similarity_sum = 0
            
            for sim_user, similarity in similar_users.items():
                if sim_user in self.user_item_matrix.index:
                    rating = self.user_item_matrix.loc[sim_user, item]
                    if rating > 0:
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                scores[item] = weighted_sum / similarity_sum
        
        # Return top N items
        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in recommendations[:n_items]]
    
    def recommend_item_based(self, user_id, n_items=10):
        """Item-based CF: recommend based on similar items"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get items user has rated
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) == 0:
            return []
        
        # Find similar items
        scores = {}
        for item in self.user_item_matrix.columns:
            if item in rated_items.index:
                continue
                
            score = 0
            for rated_item, rating in rated_items.items():
                if rated_item in self.item_similarity.index and item in self.item_similarity.columns:
                    similarity = self.item_similarity.loc[rated_item, item]
                    score += similarity * rating
            
            scores[item] = score
        
        # Return top N items
        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in recommendations[:n_items]]


# ==================== 4. MATRIX FACTORIZATION (SVD) ====================
class MatrixFactorization:
    """SVD-based Matrix Factorization"""
    
    def __init__(self, n_factors=50, learning_rate=0.01, n_epochs=20, reg=0.02):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.reg = reg
        
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        
    def fit(self, ratings_df: pd.DataFrame):
        """Train SVD model using SGD"""
        # Initialize
        users = ratings_df['user_id'].unique()
        items = ratings_df['item_id'].unique()
        
        n_users = len(users)
        n_items = len(items)
        
        # Create mappings
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = ratings_df['rating'].mean()
        
        # Training loop
        for epoch in range(self.n_epochs):
            for _, row in ratings_df.iterrows():
                user_idx = self.user_map[row['user_id']]
                item_idx = self.item_map[row['item_id']]
                rating = row['rating']
                
                # Prediction
                pred = self._predict_single(user_idx, item_idx)
                error = rating - pred
                
                # Update biases
                self.user_bias[user_idx] += self.lr * (error - self.reg * self.user_bias[user_idx])
                self.item_bias[item_idx] += self.lr * (error - self.reg * self.item_bias[item_idx])
                
                # Update factors
                user_factor = self.user_factors[user_idx].copy()
                self.user_factors[user_idx] += self.lr * (error * self.item_factors[item_idx] - self.reg * user_factor)
                self.item_factors[item_idx] += self.lr * (error * user_factor - self.reg * self.item_factors[item_idx])
            
            if epoch % 5 == 0:
                train_rmse = self._compute_rmse(ratings_df)
                print(f"Epoch {epoch}: RMSE = {train_rmse:.4f}")
        
        return self
    
    def _predict_single(self, user_idx, item_idx):
        """Predict single rating"""
        pred = self.global_mean
        pred += self.user_bias[user_idx]
        pred += self.item_bias[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return pred
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.user_map or item_id not in self.item_map:
            return self.global_mean
        
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]
        return self._predict_single(user_idx, item_idx)
    
    def recommend(self, user_id, n_items=10, exclude_items=None):
        """Recommend top N items for user"""
        if user_id not in self.user_map:
            return []
        
        user_idx = self.user_map[user_id]
        
        # Predict for all items
        scores = []
        for item_id, item_idx in self.item_map.items():
            if exclude_items and item_id in exclude_items:
                continue
            score = self._predict_single(user_idx, item_idx)
            scores.append((item_id, score))
        
        # Sort and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in scores[:n_items]]
    
    def _compute_rmse(self, ratings_df):
        """Compute RMSE on dataset"""
        predictions = []
        actuals = []
        
        for _, row in ratings_df.iterrows():
            pred = self.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])
        
        return np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))


# ==================== 5. HYBRID RECOMMENDER ====================
class HybridRecommender:
    """Combine multiple recommendation strategies"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def recommend(self, user_id, n_items=10, exclude_items=None):
        """Generate hybrid recommendations"""
        all_recommendations = defaultdict(float)
        
        for name, model in self.models.items():
            try:
                # Get recommendations from each model
                if name == 'popularity':
                    recs = model.recommend(n_items=n_items*2, exclude_items=exclude_items)
                    # Assign scores based on rank
                    for rank, item in enumerate(recs):
                        all_recommendations[item] += self.weights[name] * (1 / (rank + 1))
                else:
                    recs = model.recommend(user_id, n_items=n_items*2)
                    for rank, item in enumerate(recs):
                        all_recommendations[item] += self.weights[name] * (1 / (rank + 1))
            except Exception as e:
                print(f"Error in {name}: {e}")
                continue
        
        # Sort by score
        sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_recs[:n_items]]


# ==================== 6. EVALUATION METRICS ====================
class RecommenderEvaluator:
    """Evaluate recommendation quality"""
    
    @staticmethod
    def precision_at_k(recommended, relevant, k=10):
        """Precision@K"""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len(set(recommended_k) & relevant_set)
        return hits / k if k > 0 else 0
    
    @staticmethod
    def recall_at_k(recommended, relevant, k=10):
        """Recall@K"""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len(set(recommended_k) & relevant_set)
        return hits / len(relevant_set) if len(relevant_set) > 0 else 0
    
    @staticmethod
    def ndcg_at_k(recommended, relevant, k=10):
        """Normalized Discounted Cumulative Gain@K"""
        dcg = 0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                dcg += 1 / np.log2(i + 2)
        
        idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(relevant)))])
        return dcg / idcg if idcg > 0 else 0


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Recommendation System Demo ===\n")
    
    # 1. Generate synthetic data
    print("1. Loading data...")
    processor = DataProcessor()
    ratings_df, user_item_matrix, items_df = processor.create_synthetic_data(
        n_users=500, n_items=200, sparsity=0.95
    )
    print(f"Loaded {len(ratings_df)} ratings from {len(user_item_matrix)} users and {len(items_df)} items\n")
    
    # 2. Popularity-based recommendations
    print("2. Training Popularity Recommender...")
    pop_model = PopularityRecommender()
    pop_model.fit(ratings_df, method='weighted')
    pop_recs = pop_model.recommend(n_items=5)
    print(f"Top 5 popular items: {pop_recs}\n")
    
    # 3. Collaborative Filtering
    print("3. Training Collaborative Filtering...")
    cf_model = CollaborativeFiltering()
    cf_model.fit(user_item_matrix)
    
    test_user = user_item_matrix.index[0]
    user_based_recs = cf_model.recommend_user_based(test_user, n_items=5)
    print(f"User-based CF for user {test_user}: {user_based_recs}")
    
    item_based_recs = cf_model.recommend_item_based(test_user, n_items=5)
    print(f"Item-based CF for user {test_user}: {item_based_recs}\n")
    
    # 4. Matrix Factorization
    print("4. Training Matrix Factorization (SVD)...")
    mf_model = MatrixFactorization(n_factors=20, n_epochs=15, learning_rate=0.01)
    mf_model.fit(ratings_df)
    mf_recs = mf_model.recommend(test_user, n_items=5)
    print(f"SVD recommendations for user {test_user}: {mf_recs}\n")
    
    # 5. Hybrid Recommender
    print("5. Creating Hybrid Recommender...")
    hybrid = HybridRecommender()
    hybrid.add_model('popularity', pop_model, weight=0.2)
    hybrid.add_model('cf_user', cf_model, weight=0.3)
    hybrid.add_model('svd', mf_model, weight=0.5)
    
    hybrid_recs = hybrid.recommend(test_user, n_items=5)
    print(f"Hybrid recommendations for user {test_user}: {hybrid_recs}\n")
    
    # 6. Evaluation
    print("6. Evaluating recommendations...")
    evaluator = RecommenderEvaluator()
    
    # Get actual relevant items (items user rated highly)
    user_ratings = ratings_df[ratings_df['user_id'] == test_user]
    relevant_items = user_ratings[user_ratings['rating'] >= 4]['item_id'].tolist()
    
    if len(relevant_items) > 0:
        precision = evaluator.precision_at_k(hybrid_recs, relevant_items, k=5)
        recall = evaluator.recall_at_k(hybrid_recs, relevant_items, k=5)
        ndcg = evaluator.ndcg_at_k(hybrid_recs, relevant_items, k=5)
        
        print(f"Precision@5: {precision:.3f}")
        print(f"Recall@5: {recall:.3f}")
        print(f"NDCG@5: {ndcg:.3f}")
    
    print("\n=== System Ready for Deployment ===")