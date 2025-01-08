import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import re
import kagglehub
import os
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import pickle
from scipy import sparsez
from annoy import AnnoyIndex

class RecipeRecommender:
    def __init__(self, cache_dir: str = "recipe_data"):
        """
        Initialize the RecipeRecommender with necessary components.
        
        Args:
            cache_dir (str): Directory to cache downloaded dataset and preprocessed data
        """
        self.recipes_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.user_preferences = {}
        self.cache_dir = cache_dir
        self.annoy_index = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_dataset(self) -> str:
        """
        Use local dataset or download the RecipeNLG dataset from Kaggle.
        
        Returns:
            str: Path to the dataset
        """
        try:
            # Check if dataset exists in recipe_data
            cache_path = os.path.join(self.cache_dir, "full_dataset.csv")
            if os.path.exists(cache_path):
                print("Using existing dataset from recipe_data")
                return cache_path
            
            # If file doesn't exist, raise error since user expects it to be there
            raise FileNotFoundError(f"Expected dataset not found at: {cache_path}")
            
        except Exception as e:
            print(f"Error accessing dataset: {e}")
            raise

    def _parse_list_string(self, list_str: str) -> List[str]:
        """Parse string representation of lists into actual lists."""
        try:
            if isinstance(list_str, str):
                # Remove any leading/trailing whitespace and convert to lowercase
                cleaned_str = list_str.strip().lower()
                # Parse the string as a literal Python expression
                parsed_list = ast.literal_eval(cleaned_str)
                # Clean each item in the list
                return [str(item).strip(' "\'') for item in parsed_list]
            return []
        except:
            return []

    def load_and_preprocess_data(self, filepath: Optional[str] = None) -> None:
        """Load and preprocess the recipe dataset."""
        try:
            # Check for cached preprocessed data
            cached_data_path = os.path.join(self.cache_dir, "preprocessed_data.pkl")
            cached_tfidf_path = os.path.join(self.cache_dir, "tfidf_data.npz")
            cached_vectorizer_path = os.path.join(self.cache_dir, "tfidf_vectorizer.pkl")

            if os.path.exists(cached_data_path) and os.path.exists(cached_tfidf_path) and os.path.exists(cached_vectorizer_path):
                print("Loading cached preprocessed data...")
                self.recipes_df = pd.read_pickle(cached_data_path)
                self.tfidf_matrix = np.load(cached_tfidf_path, allow_pickle=True)['matrix']
                with open(cached_vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                print(f"Successfully loaded cached data with {len(self.recipes_df)} recipes")
                return

            # If no cached data, proceed with normal preprocessing
            if filepath is None:
                filepath = self.download_dataset()
                
            print(f"Loading dataset from {filepath}...")
            self.recipes_df = pd.read_csv(filepath)
            
            print("Preprocessing data...")
            # Clean ingredients and NER columns
            self.recipes_df['clean_ingredients'] = self.recipes_df['ingredients'].apply(
                lambda x: ' '.join(self._parse_list_string(x))
            )
            
            if 'NER' in self.recipes_df.columns:
                self.recipes_df['parsed_ner'] = self.recipes_df['NER'].apply(
                    self._parse_list_string
                )
            
            # Create TF-IDF vectors for ingredients
            print("Creating TF-IDF vectors...")
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.recipes_df['clean_ingredients']
            )
            
            # Cache the preprocessed data
            print("Caching preprocessed data...")
            self.recipes_df.to_pickle(cached_data_path)
            np.savez_compressed(cached_tfidf_path, matrix=self.tfidf_matrix)
            with open(cached_vectorizer_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            print(f"Successfully loaded and preprocessed {len(self.recipes_df)} recipes")
            
        except Exception as e:
            print(f"Error loading and preprocessing data: {e}")
            raise

    def set_user_preferences(self, preferences: Dict) -> None:
        """Set user preferences for recipe recommendations."""
        self.user_preferences = preferences
        print("User preferences updated:", preferences)

    def filter_recipes(self) -> pd.DataFrame:
        """Filter recipes based on user preferences using available data."""
        filtered_df = self.recipes_df.copy()
        
        # Enhanced cuisine keywords dictionary
        cuisine_keywords = {
            'indian': [
                'curry', 'masala', 'tikka', 'dal', 'paneer', 'naan', 'biryani', 
                'chutney', 'korma', 'tandoori', 'raita', 'samosa', 'garam masala',
                'turmeric', 'cumin', 'coriander'
            ],
            'chinese': [
                'soy sauce', 'ginger', 'stir fry', 'wok', 'chinese', 'sesame oil',
                'hoisin', 'szechuan', 'tofu', 'bok choy', 'chow mein', 'dim sum'
            ],
            'italian': [
                'pasta', 'pizza', 'risotto', 'italian', 'pesto', 'parmesan',
                'mozzarella', 'marinara', 'lasagna', 'cannelloni', 'ravioli',
                'bruschetta', 'focaccia'
            ],
            'mexican': [
                'taco', 'burrito', 'salsa', 'mexican', 'enchilada', 'tortilla',
                'guacamole', 'quesadilla', 'fajita', 'chipotle', 'jalapeño',
                'cilantro', 'queso'
            ],
            'japanese': [
                'sushi', 'miso', 'ramen', 'udon', 'teriyaki', 'wasabi',
                'tempura', 'sashimi', 'dashi', 'sake', 'mirin'
            ],
            'thai': [
                'thai', 'curry paste', 'coconut milk', 'fish sauce', 'lemongrass',
                'pad thai', 'satay', 'green curry', 'red curry', 'thai basil'
            ]
        }
        
        # Filter for dietary restrictions
        if 'dietary_restrictions' in self.user_preferences:
            for restriction in self.user_preferences['dietary_restrictions']:
                if restriction.lower() == 'vegetarian':
                    meat_terms = [
                        'chicken', 'beef', 'pork', 'meat', 'fish', 'lamb', 'turkey',
                        'bacon', 'prosciutto', 'ham', 'seafood', 'shrimp', 'duck',
                        'veal', 'anchovies'
                    ]
                    for term in meat_terms:
                        filtered_df = filtered_df[
                            ~filtered_df['clean_ingredients'].str.contains(
                                term, 
                                case=False, 
                                na=False
                            )
                        ]
                elif restriction.lower() == 'vegan':
                    animal_products = [
                        'meat', 'fish', 'chicken', 'beef', 'pork', 'lamb', 'turkey',
                        'egg', 'milk', 'cream', 'cheese', 'yogurt', 'butter', 'honey',
                        'mayo', 'mayonnaise', 'gelatin', 'whey', 'casein'
                    ]
                    for term in animal_products:
                        filtered_df = filtered_df[
                            ~filtered_df['clean_ingredients'].str.contains(
                                term, 
                                case=False, 
                                na=False
                            )
                        ]
        
        # Enhanced cuisine filtering using both ingredients and NER
        if 'cuisine_preference' in self.user_preferences:
            cuisine = self.user_preferences['cuisine_preference'].lower()
            if cuisine in cuisine_keywords:
                cuisine_pattern = '|'.join(cuisine_keywords[cuisine])
                
                # Create mask using multiple columns
                title_mask = filtered_df['title'].str.lower().str.contains(cuisine_pattern, na=False)
                ingredients_mask = filtered_df['clean_ingredients'].str.lower().str.contains(cuisine_pattern, na=False)
                ner_mask = filtered_df['NER'].str.lower().str.contains(cuisine_pattern, na=False)
                
                # Combine masks
                filtered_df = filtered_df[title_mask | ingredients_mask | ner_mask]
        
        # Filter out excluded ingredients
        if 'excluded_ingredients' in self.user_preferences:
            for ingredient in self.user_preferences['excluded_ingredients']:
                filtered_df = filtered_df[
                    ~filtered_df['clean_ingredients'].str.contains(
                        ingredient.lower(), 
                        case=False, 
                        na=False
                    )
                ]
        
        return filtered_df

    def build_annoy_index(self):
        """Build Annoy index for fast similarity search"""
        n_features = self.tfidf_matrix.shape[1]
        self.annoy_index = AnnoyIndex(n_features, 'angular')  # angular distance = cosine similarity
        
        # Add items to index
        for i in range(self.tfidf_matrix.shape[0]):
            self.annoy_index.add_item(i, self.tfidf_matrix[i].toarray()[0])
        
        # Build the index
        self.annoy_index.build(10)  # 10 trees - more trees = better accuracy but slower build
        
    def get_ingredient_similarity(self, available_ingredients: List[str], n_recipes: int = 5) -> List[Dict]:
        """Find recipes similar to available ingredients using Annoy"""
        ingredients_text = ' '.join(available_ingredients)
        ingredients_vector = self.tfidf_vectorizer.transform([ingredients_text]).toarray()[0]
        
        # Get approximate nearest neighbors
        similar_indices = self.annoy_index.get_nns_by_vector(
            ingredients_vector, 
            n_recipes, 
            include_distances=True
        )
        
        recommendations = []
        for idx, distance in zip(similar_indices[0], similar_indices[1]):
            similarity_score = 1 - (distance ** 2) / 2  # Convert angular distance to cosine similarity
            rec = {
                'title': self.recipes_df.iloc[idx]['title'],
                'similarity_score': similarity_score,
                'ingredients': self.recipes_df.iloc[idx]['ingredients'],
            }
            
            for field in ['directions', 'link', 'source']:
                if field in self.recipes_df.columns:
                    rec[field] = self.recipes_df.iloc[idx][field]
            
            recommendations.append(rec)
            
        return recommendations

    def recommend_recipes(self, 
                         available_ingredients: List[str], 
                         n_recommendations: int = 5) -> List[Dict]:
        """Generate recipe recommendations based on user preferences and available ingredients."""
        # Filter recipes based on user preferences
        filtered_recipes_df = self.filter_recipes()
        
        if filtered_recipes_df.empty:
            print("No recipes found matching the current preferences.")
            return []
        
        # Create a temporary recommender with only filtered recipes
        temp_recommender = RecipeRecommender()
        temp_recommender.recipes_df = filtered_recipes_df
        temp_recommender.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        temp_recommender.tfidf_matrix = temp_recommender.tfidf_vectorizer.fit_transform(
            filtered_recipes_df['clean_ingredients']
        )
        
        # Build the Annoy index before using it
        temp_recommender.build_annoy_index()
        
        # Get recommendations from filtered dataset
        recommendations = temp_recommender.get_ingredient_similarity(
            available_ingredients, 
            n_recommendations
        )
        
        return recommendations

def print_recommendations(recommendations: List[Dict], output_file: str = "recipe_recommendations.txt") -> None:
    """
    Helper function to print recommendations in a readable format and save to file.
    Creates uniquely named files by appending numbers if file already exists.
    
    Args:
        recommendations: List of recipe recommendations
        output_file: Base path to save the recommendations (default: recipe_recommendations.txt)
    """
    if not recommendations:
        message = "No recommendations found."
        print(message)
        with open(output_file, 'w') as f:
            f.write(message)
        return
    
    # Create unique filename
    base_path = Path(output_file)
    counter = 1
    new_path = base_path
    while new_path.exists():
        new_path = base_path.parent / f"{base_path.stem}_{counter}{base_path.suffix}"
        counter += 1
    
    # Create output text
    output = []
    for i, rec in enumerate(recommendations, 1):
        output.append(f"\nRecipe {i}: {rec['title']}")
        output.append(f"Match Score: {rec['similarity_score']:.2f}")
        
        # Parse and clean ingredients list
        ingredients = ast.literal_eval(rec['ingredients']) if isinstance(rec['ingredients'], str) else rec['ingredients']
        output.append("\nIngredients:")
        for ingredient in ingredients:
            output.append(f"- {ingredient}")
        
        if 'directions' in rec:
            output.append("\nDirections:")
            directions = ast.literal_eval(rec['directions']) if isinstance(rec['directions'], str) else rec['directions']
            for j, step in enumerate(directions, 1):
                output.append(f"{j}. {step}")
        
        if 'source' in rec:
            output.append(f"\nSource: {rec['source']}")
        if 'link' in rec:
            output.append(f"Link: {rec['link']}")
        output.append("-" * 50)
    
    # Join all lines with newlines
    output_text = '\n'.join(output)
    
    # Print to console
    print(output_text)
    
    # Save to file using the unique path
    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"\nRecommendations saved to: {new_path}")

def main():
    """Main function to run the recipe recommender."""
    try:
        # Initialize recommender
        print("Initializing Recipe Recommender...")
        recommender = RecipeRecommender()
        
        # Load and preprocess data
        recommender.load_and_preprocess_data()
        
        # Example user preferences (can be modified)
        user_preferences = {
            'dietary_restrictions': ['vegetarian', 'gluten-free'],
            'cuisine_preference': 'thai',
            'excluded_ingredients': ['peanuts']
        }
        recommender.set_user_preferences(user_preferences)
        
        # Example available ingredients (can be modified)
        available_ingredients = ['coconut milk', 'tofu', 'vegetables', 'lime']
        print("\nFinding recipes with ingredients:", available_ingredients)
        
        # Get recommendations
        recommendations = recommender.recommend_recipes(available_ingredients)
        
        # Print recommendations
        print_recommendations(recommendations)
        
    except Exception as e:
        print(f"Error in recipe recommendation system: {e}")
        raise

if __name__ == "__main__":
    main()