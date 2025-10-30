"""
Train and save recommendation models
"""
import pickle
from recommender import (
    DataProcessor, PopularityRecommender, 
    CollaborativeFiltering, MatrixFactorization, HybridRecommender
)

print("Training Recommendation Models...")

# Load/Generate data
processor = DataProcessor()
ratings_df, user_item_matrix, items_df = processor.create_synthetic_data(
    n_users=1000, n_items=500, sparsity=0.95
)

print(f"Dataset: {len(ratings_df)} ratings")

# Train Popularity Model
print("Training Popularity Model...")
pop_model = PopularityRecommender()
pop_model.fit(ratings_df, method='weighted')

# Train Collaborative Filtering
print("Training Collaborative Filtering...")
cf_model = CollaborativeFiltering()
cf_model.fit(user_item_matrix)

# Train Matrix Factorization
print("Training Matrix Factorization (SVD)...")
mf_model = MatrixFactorization(n_factors=30, n_epochs=20)
mf_model.fit(ratings_df)

# Create Hybrid Model
print("Creating Hybrid Model...")
hybrid = HybridRecommender()
hybrid.add_model('popularity', pop_model, weight=0.2)
hybrid.add_model('cf', cf_model, weight=0.3)
hybrid.add_model('svd', mf_model, weight=0.5)

# Save models
print("Saving models...")
import os
os.makedirs('models', exist_ok=True)

with open('models/popularity.pkl', 'wb') as f:
    pickle.dump(pop_model, f)

with open('models/collaborative.pkl', 'wb') as f:
    pickle.dump(cf_model, f)

with open('models/svd.pkl', 'wb') as f:
    pickle.dump(mf_model, f)

with open('models/hybrid.pkl', 'wb') as f:
    pickle.dump(hybrid, f)

print("✅ Models trained and saved successfully!")