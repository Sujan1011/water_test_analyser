import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class ColorClassifier:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.color_classes = {
            0: {'name': 'red', 'ph_range': (0, 3), 'iron_range': (3, 5), 'nitrate_range': (50, 100), 'chlorine_range': (7, 10)},
            1: {'name': 'orange', 'ph_range': (3, 5), 'iron_range': (1, 3), 'nitrate_range': (20, 50), 'chlorine_range': (5, 7)},
            2: {'name': 'yellow', 'ph_range': (5, 6.5), 'iron_range': (0.3, 1), 'nitrate_range': (10, 20), 'chlorine_range': (3, 5)},
            3: {'name': 'green', 'ph_range': (6.5, 7.5), 'iron_range': (0.1, 0.3), 'nitrate_range': (5, 10), 'chlorine_range': (2, 3)},
            4: {'name': 'blue', 'ph_range': (7.5, 8.5), 'iron_range': (0, 0.1), 'nitrate_range': (1, 5), 'chlorine_range': (1, 2)},
            5: {'name': 'purple', 'ph_range': (8.5, 14), 'iron_range': (0, 0), 'nitrate_range': (0, 1), 'chlorine_range': (0, 1)},
            6: {'name': 'light_yellow', 'ph_range': (6, 7), 'iron_range': (0, 0.1), 'nitrate_range': (0, 1), 'chlorine_range': (0.5, 1.0)},
            7: {'name': 'golden_yellow', 'ph_range': (5, 6), 'iron_range': (0.1, 0.3), 'nitrate_range': (1, 5), 'chlorine_range': (3, 5)},
            8: {'name': 'deep_yellow', 'ph_range': (4, 5), 'iron_range': (0.3, 1), 'nitrate_range': (5, 10), 'chlorine_range': (5, 7)},
            9: {'name': 'pink', 'ph_range': (7, 8), 'iron_range': (0, 0.1), 'nitrate_range': (1, 5), 'chlorine_range': (0, 0.5)},
            10: {'name': 'dark_pink', 'ph_range': (8, 9), 'iron_range': (0, 0), 'nitrate_range': (10, 20), 'chlorine_range': (0, 0)},
            11: {'name': 'brown', 'ph_range': (3, 4), 'iron_range': (5, 10), 'nitrate_range': (50, 100), 'chlorine_range': (0, 0)}
        }
    
    def extract_features(self, rgb_values):
        """Extract features from RGB values for classification"""
        r, g, b = rgb_values
        
        # Normalize RGB values
        total = r + g + b
        if total > 0:
            r_norm, g_norm, b_norm = r/total, g/total, b/total
        else:
            r_norm, g_norm, b_norm = 0, 0, 0
        
        # Calculate additional features
        features = [
            r, g, b,  # Raw RGB
            r_norm, g_norm, b_norm,  # Normalized RGB
            r - g, r - b, g - b,  # Color differences
            (r + g + b) / 3,  # Brightness
            max(r, g, b) - min(r, g, b),  # Color saturation
            np.arctan2(np.sqrt(3)*(g - b), 2*r - g - b),  # Hue angle approximation
            r * g, r * b, g * b,  # Color interactions
            np.sqrt(r**2 + g**2 + b**2),  # Color magnitude
            np.std([r, g, b]),  # Color variance
            np.median([r, g, b])  # Color median
        ]
        
        return np.array(features)
    
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data for color classification"""
        X = []
        y = []
        
        for class_id, class_info in self.color_classes.items():
            for _ in range(n_samples):
                # Generate RGB values for this color class with variation
                if class_info['name'] == 'red':
                    r = np.random.uniform(180, 255)
                    g = np.random.uniform(0, 80)
                    b = np.random.uniform(0, 80)
                elif class_info['name'] == 'orange':
                    r = np.random.uniform(200, 255)
                    g = np.random.uniform(100, 180)
                    b = np.random.uniform(0, 50)
                elif class_info['name'] == 'yellow':
                    r = np.random.uniform(200, 255)
                    g = np.random.uniform(200, 255)
                    b = np.random.uniform(0, 100)
                elif class_info['name'] == 'green':
                    r = np.random.uniform(0, 150)
                    g = np.random.uniform(150, 255)
                    b = np.random.uniform(0, 150)
                elif class_info['name'] == 'blue':
                    r = np.random.uniform(0, 100)
                    g = np.random.uniform(0, 150)
                    b = np.random.uniform(150, 255)
                elif class_info['name'] == 'purple':
                    r = np.random.uniform(150, 255)
                    g = np.random.uniform(0, 100)
                    b = np.random.uniform(150, 255)
                elif class_info['name'] == 'light_yellow':
                    r = np.random.uniform(240, 255)
                    g = np.random.uniform(240, 255)
                    b = np.random.uniform(200, 240)
                elif class_info['name'] == 'golden_yellow':
                    r = np.random.uniform(220, 255)
                    g = np.random.uniform(180, 220)
                    b = np.random.uniform(0, 80)
                elif class_info['name'] == 'deep_yellow':
                    r = np.random.uniform(200, 240)
                    g = np.random.uniform(160, 200)
                    b = np.random.uniform(0, 60)
                elif class_info['name'] == 'pink':
                    r = np.random.uniform(220, 255)
                    g = np.random.uniform(150, 200)
                    b = np.random.uniform(150, 200)
                elif class_info['name'] == 'dark_pink':
                    r = np.random.uniform(200, 240)
                    g = np.random.uniform(100, 150)
                    b = np.random.uniform(100, 150)
                elif class_info['name'] == 'brown':
                    r = np.random.uniform(100, 180)
                    g = np.random.uniform(50, 120)
                    b = np.random.uniform(20, 80)
                else:
                    r = np.random.uniform(0, 255)
                    g = np.random.uniform(0, 255)
                    b = np.random.uniform(0, 255)
                
                # Add noise
                r += np.random.normal(0, 15)
                g += np.random.normal(0, 15)
                b += np.random.normal(0, 15)
                
                # Clip to valid range
                r = np.clip(r, 0, 255)
                g = np.clip(g, 0, 255)
                b = np.clip(b, 0, 255)
                
                features = self.extract_features([r, g, b])
                X.append(features)
                y.append(class_id)
        
        return np.array(X), np.array(y)
    
    def train(self):
        """Train the classifier"""
        print("Generating training data...")
        X, y = self.generate_training_data(n_samples=1500)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
    
    def predict_color(self, rgb_values):
        """Predict color class from RGB values"""
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        features = self.extract_features(rgb_values)
        features_scaled = self.scaler.transform([features])
        
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'color_name': self.color_classes[idx]['name'],
                'probability': float(probabilities[idx])
            }
            for idx in top_indices
        ]
        
        return {
            'class_id': int(prediction),
            'color_name': self.color_classes[prediction]['name'],
            'confidence': float(probabilities[prediction]),
            'top_predictions': top_predictions,
            'all_probabilities': {
                self.color_classes[i]['name']: float(probabilities[i])
                for i in range(len(probabilities))
            }
        }
    
    def predict_batch(self, rgb_list):
        """Predict colors for multiple RGB values"""
        features_list = [self.extract_features(rgb) for rgb in rgb_list]
        features_scaled = self.scaler.transform(features_list)
        
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                'class_id': int(pred),
                'color_name': self.color_classes[pred]['name'],
                'confidence': float(probs[pred]),
                'rgb': rgb_list[i]
            })
        
        return results
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'color_classes': self.color_classes
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        model_data = joblib.load(filepath)
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.color_classes = model_data['color_classes']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Create and train classifier
    classifier = ColorClassifier()
    classifier.train()
    
    # Test prediction for all color types
    test_colors = [
        ([255, 50, 50], "Red"),
        ([255, 165, 0], "Orange"),
        ([255, 255, 0], "Yellow"),
        ([0, 255, 0], "Green"),
        ([0, 0, 255], "Blue"),
        ([128, 0, 128], "Purple"),
        ([255, 255, 200], "Light Yellow"),
        ([255, 215, 0], "Golden Yellow"),
        ([255, 140, 0], "Deep Yellow"),
        ([255, 192, 203], "Pink"),
        ([255, 105, 180], "Dark Pink"),
        ([139, 69, 19], "Brown")
    ]
    
    for rgb, expected in test_colors:
        result = classifier.predict_color(rgb)
        print(f"\nExpected: {expected}")
        print(f"Predicted: {result['color_name']} (confidence: {result['confidence']:.3f})")
        # Fixed the nested quotes issue here
        top_predictions_str = [(p['color_name'], f"{p['probability']:.3f}") for p in result['top_predictions']]
        print(f"Top predictions: {top_predictions_str}")
    
    # Save model
    classifier.save_model("color_classifier.pkl")