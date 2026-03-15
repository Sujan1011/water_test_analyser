from flask import Flask, render_template, request, jsonify, send_file, Response
import cv2
import numpy as np
import base64
from PIL import Image
import io
import json
from datetime import datetime
import os
import sys
import csv
from io import StringIO

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_processor import ImageProcessor
from ml_classifier import ColorClassifier

app = Flask(__name__)

class WaterTestWebApp:
    def __init__(self):
        self.test_results = []
        self.image_processor = ImageProcessor()
        self.ml_classifier = ColorClassifier()
        
        # Try to load pre-trained model
        self.initialize_classifier()
        
        # Color reference database
        self.color_references = self.load_color_references()
        
    def initialize_classifier(self):
        """Initialize the ML classifier"""
        try:
            if os.path.exists("color_classifier.pkl"):
                self.ml_classifier.load_model("color_classifier.pkl")
                print("Loaded pre-trained classifier")
        except Exception as e:
            print(f"Classifier initialization warning: {e}")
    
    def load_color_references(self):
        """Load color reference database"""
        return {
            'ph': {
                'red': {'range': (0, 3), 'color': 'Red', 'level': 'Strong Acid', 'value': 2.0, 'unit': 'pH'},
                'orange': {'range': (3, 5), 'color': 'Orange', 'level': 'Acidic', 'value': 4.0, 'unit': 'pH'},
                'yellow': {'range': (5, 6.5), 'color': 'Yellow', 'level': 'Slightly Acidic', 'value': 6.0, 'unit': 'pH'},
                'green': {'range': (6.5, 7.5), 'color': 'Green', 'level': 'Neutral', 'value': 7.0, 'unit': 'pH'},
                'blue': {'range': (7.5, 8.5), 'color': 'Blue', 'level': 'Slightly Basic', 'value': 8.0, 'unit': 'pH'},
                'purple': {'range': (8.5, 14), 'color': 'Purple', 'level': 'Basic', 'value': 9.0, 'unit': 'pH'}
            },
            'iron': {
                'very_light_yellow': {'range': (0, 0.05), 'color': 'Very Light Yellow', 'level': 'None', 'value': 0.02, 'unit': 'mg/L'},
                'light_yellow': {'range': (0.05, 0.1), 'color': 'Light Yellow', 'level': 'None', 'value': 0.05, 'unit': 'mg/L'},
                'pale_yellow': {'range': (0.1, 0.15), 'color': 'Pale Yellow', 'level': 'Trace', 'value': 0.1, 'unit': 'mg/L'},
                'yellow': {'range': (0.15, 0.3), 'color': 'Yellow', 'level': 'Low', 'value': 0.2, 'unit': 'mg/L'},
                'golden_yellow': {'range': (0.3, 0.5), 'color': 'Golden Yellow', 'level': 'Low-Moderate', 'value': 0.4, 'unit': 'mg/L'},
                'orange_yellow': {'range': (0.5, 0.8), 'color': 'Orange-Yellow', 'level': 'Moderate', 'value': 0.6, 'unit': 'mg/L'},
                'orange': {'range': (0.8, 1.2), 'color': 'Orange', 'level': 'Moderate', 'value': 1.0, 'unit': 'mg/L'},
                'deep_orange': {'range': (1.2, 1.8), 'color': 'Deep Orange', 'level': 'Moderate-High', 'value': 1.5, 'unit': 'mg/L'},
                'red_orange': {'range': (1.8, 2.5), 'color': 'Red-Orange', 'level': 'High', 'value': 2.0, 'unit': 'mg/L'},
                'red_brown': {'range': (2.5, 4.0), 'color': 'Red-Brown', 'level': 'Very High', 'value': 3.0, 'unit': 'mg/L'},
                'brown': {'range': (4.0, 6.0), 'color': 'Brown', 'level': 'Severe', 'value': 5.0, 'unit': 'mg/L'},
                'dark_brown': {'range': (6.0, 10), 'color': 'Dark Brown', 'level': 'Severe', 'value': 7.0, 'unit': 'mg/L'}
            },
            'hardness': {
                'light_blue': {'range': (0, 50), 'color': 'Light Blue', 'level': 'Very Soft', 'value': 25, 'unit': 'mg/L'},
                'blue': {'range': (50, 100), 'color': 'Blue', 'level': 'Soft', 'value': 75, 'unit': 'mg/L'},
                'blue_green': {'range': (100, 150), 'color': 'Blue-Green', 'level': 'Slightly Hard', 'value': 125, 'unit': 'mg/L'},
                'green': {'range': (150, 200), 'color': 'Green', 'level': 'Moderately Hard', 'value': 175, 'unit': 'mg/L'},
                'green_yellow': {'range': (200, 250), 'color': 'Green-Yellow', 'level': 'Hard', 'value': 225, 'unit': 'mg/L'},
                'yellow': {'range': (250, 300), 'color': 'Yellow', 'level': 'Very Hard', 'value': 275, 'unit': 'mg/L'},
                'orange': {'range': (300, 400), 'color': 'Orange', 'level': 'Extremely Hard', 'value': 350, 'unit': 'mg/L'},
                'red': {'range': (400, 500), 'color': 'Red', 'level': 'Severe', 'value': 450, 'unit': 'mg/L'}
            },
            'chlorine': {
                'very_light_yellow': {'range': (0, 0.5), 'color': 'Very Light Yellow', 'level': 'Very Low', 'value': 0.2, 'unit': 'mg/L'},
                'light_yellow': {'range': (0.5, 1.0), 'color': 'Light Yellow', 'level': 'Low', 'value': 0.8, 'unit': 'mg/L'},
                'pale_yellow': {'range': (1.0, 2.0), 'color': 'Pale Yellow', 'level': 'Low-Moderate', 'value': 1.5, 'unit': 'mg/L'},
                'yellow': {'range': (2.0, 3.0), 'color': 'Yellow', 'level': 'Moderate', 'value': 2.5, 'unit': 'mg/L'},
                'golden_yellow': {'range': (3.0, 5.0), 'color': 'Golden Yellow', 'level': 'Moderate-High', 'value': 4.0, 'unit': 'mg/L'},
                'deep_yellow': {'range': (5.0, 7.0), 'color': 'Deep Yellow', 'level': 'High', 'value': 6.0, 'unit': 'mg/L'},
                'orange_yellow': {'range': (7.0, 10.0), 'color': 'Orange-Yellow', 'level': 'Very High', 'value': 8.5, 'unit': 'mg/L'},
                'orange': {'range': (7.0, 10.0), 'color': 'Orange', 'level': 'Very High', 'value': 8.5, 'unit': 'mg/L'}
            }
        }
        
    def process_image(self, image_data):
        """Process base64 encoded image"""
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Save temporarily for processing
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, opencv_image)
        
        # Process with image processor
        processed = self.image_processor.preprocess_image(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return processed, opencv_image
    
    def analyze_color(self, processed_image, test_type):
        """Analyze color from image using ML and image processing"""
        if processed_image is None:
            return None
        
        # Detect test strip region
        strip_region = self.image_processor.detect_test_strip(processed_image)
        
        # Extract test region
        test_region = self.image_processor.extract_test_region(processed_image, strip_region)
        
        # Analyze color distribution
        color_distribution = self.image_processor.analyze_color_distribution(test_region)
        
        # Get dominant color
        dominant_color = color_distribution[0]['rgb'] if color_distribution else [128, 128, 128]
        
        # Get color name
        color_name = self.rgb_to_color_name(dominant_color, test_type)
        
        # Get ML prediction if available
        ml_result = None
        if self.ml_classifier.is_trained:
            try:
                ml_result = self.ml_classifier.predict_color(dominant_color)
            except:
                pass
        
        # Get color statistics
        statistics = self.image_processor.calculate_color_statistics(test_region)
        
        # Determine result based on test type
        result = self.determine_result(color_name, dominant_color, test_type)
        
        return {
            'color_name': color_name,
            'rgb': dominant_color.tolist() if isinstance(dominant_color, np.ndarray) else dominant_color,
            'hex': '#{:02x}{:02x}{:02x}'.format(int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])),
            'color_distribution': color_distribution,
            'statistics': statistics,
            'result': result,
            'ml_analysis': ml_result
        }
    
    def rgb_to_color_name(self, rgb, test_type=None):
        """Convert RGB to color name with enhanced detection for all shades"""
        r, g, b = rgb
        
        # Print RGB values for debugging
        print(f"Detected RGB: R={r:.0f}, G={g:.0f}, B={b:.0f}")
        
        # Calculate ratios for better detection
        total = r + g + b
        if total > 0:
            r_ratio = r / total
            g_ratio = g / total
            b_ratio = b / total
            
            print(f"RGB Ratios - R:{r_ratio:.3f}, G:{g_ratio:.3f}, B:{b_ratio:.3f}")
            
            # ============ HARDNESS TEST SPECIFIC DETECTION ============
            if test_type == 'hardness':
                # Light Blue / Very Soft (0-50 mg/L)
                if b > 200 and g > 150 and r < 150:
                    if b > 230 and g > 180:
                        return 'light_blue'
                    else:
                        return 'blue'
                
                # Blue / Soft (50-100 mg/L)
                if b > 180 and g > 120 and g < 180 and r < 120:
                    return 'blue'
                
                # Blue-Green / Slightly Hard (100-150 mg/L)
                if b > 150 and g > 150 and r < 150:
                    if g > b:
                        return 'blue_green'
                    else:
                        return 'blue'
                
                # Green / Moderately Hard (150-200 mg/L)
                if g > 180 and r < 150 and b < 150:
                    if g > 200:
                        return 'green'
                    else:
                        return 'blue_green'
                
                # Green-Yellow / Hard (200-250 mg/L)
                if g > 180 and r > 150 and r < 200 and b < 120:
                    return 'green_yellow'
                
                # Yellow / Very Hard (250-300 mg/L)
                if r > 200 and g > 180 and b < 100:
                    return 'yellow'
                
                # Orange / Extremely Hard (300-400 mg/L)
                if r > 210 and g > 120 and g < 180 and b < 80:
                    return 'orange'
                
                # Red / Severe (400+ mg/L)
                if r > 200 and g < 120 and b < 80:
                    return 'red'
            
            # ============ IRON TEST SPECIFIC DETECTION ============
            if test_type == 'iron':
                # Very Light Yellow / None (0.02 mg/L) - Almost white with slight yellow
                if r > 245 and g > 245 and b > 240:
                    return 'very_light_yellow'
                
                # Light Yellow / None (0.05 mg/L)
                if r > 240 and g > 240 and b > 220:
                    return 'light_yellow'
                
                # Pale Yellow / Trace (0.1 mg/L)
                if r > 230 and g > 230 and b > 200:
                    return 'pale_yellow'
                
                # Yellow / Low (0.2 mg/L)
                if r > 220 and g > 210 and b < 180:
                    if b < 150:
                        return 'yellow'
                    else:
                        return 'pale_yellow'
                
                # Golden Yellow / Low-Moderate (0.4 mg/L)
                if r > 220 and g > 190 and b < 120:
                    if r > 235 and g > 210:
                        return 'golden_yellow'
                    else:
                        return 'yellow'
                
                # Orange-Yellow / Moderate (0.6 mg/L)
                if r > 210 and g > 160 and g < 200 and b < 90:
                    return 'orange_yellow'
                
                # Orange / Moderate (1.0 mg/L)
                if r > 210 and g > 120 and g < 170 and b < 70:
                    return 'orange'
                
                # Deep Orange / Moderate-High (1.5 mg/L)
                if r > 200 and g > 90 and g < 140 and b < 50:
                    return 'deep_orange'
                
                # Red-Orange / High (2.0 mg/L)
                if r > 190 and g > 60 and g < 110 and b < 40:
                    return 'red_orange'
                
                # Red-Brown / Very High (3.0 mg/L)
                if r > 160 and r < 210 and g > 40 and g < 80 and b < 35:
                    return 'red_brown'
                
                # Brown / Severe (5.0 mg/L)
                if r > 120 and r < 180 and g > 30 and g < 70 and b < 30:
                    return 'brown'
                
                # Dark Brown / Severe (7.0 mg/L)
                if r > 80 and r < 150 and g > 20 and g < 60 and b < 25:
                    return 'dark_brown'
            
            # ============ CHLORINE TEST DETECTION ============
            if test_type == 'chlorine':
                # Yellow detection - high red and green, low blue
                if r > 180 and g > 150 and b < 120:
                    if b < 50:
                        if r > 230 and g > 210:
                            return 'very_light_yellow'
                        elif r > 210 and g > 190:
                            return 'light_yellow'
                        elif r > 200 and g > 170:
                            return 'pale_yellow'
                        else:
                            return 'yellow'
                    elif b < 80:
                        if r > 220 and g > 190:
                            return 'golden_yellow'
                        else:
                            return 'deep_yellow'
                    elif b < 100:
                        return 'orange_yellow'
                
                # Orange detection
                if r > 200 and g > 100 and g < 180 and b < 80:
                    if g > 150:
                        return 'orange_yellow'
                    else:
                        return 'orange'
            
            # ============ PH TEST DETECTION ============
            if test_type == 'ph':
                # Red detection
                if r > 200 and g < 100 and b < 100:
                    return 'red'
                
                # Orange detection
                if r > 200 and g > 100 and g < 180 and b < 80:
                    return 'orange'
                
                # Yellow detection
                if r > 200 and g > 180 and b < 120:
                    return 'yellow'
                
                # Green detection
                if r < 150 and g > 150 and b < 150:
                    return 'green'
                
                # Blue detection
                if r < 150 and g < 150 and b > 150:
                    return 'blue'
                
                # Purple detection
                if r > 150 and g < 150 and b > 150:
                    return 'purple'
        
        # Default color classification for any test
        # First check for white/light colors
        if r > 220 and g > 220 and b > 220:
            return 'white'
        
        # Check for black/dark colors
        if r < 50 and g < 50 and b < 50:
            return 'black'
        
        # Red-orange detection
        if r > 200 and g > 80 and g < 150 and b < 50:
            return 'red_orange'
        
        # Red detection
        if r > 200 and g < 100 and b < 100:
            return 'red'
        
        # Blue detection
        if r < 150 and g < 150 and b > 150:
            return 'blue'
        
        # Green detection
        if r < 150 and g > 150 and b < 150:
            return 'green'
        
        # Brown detection
        if r > 100 and r < 180 and g > 50 and g < 120 and b < 80:
            return 'brown'
        
        # Default: try to find closest by max value
        max_val = max(r, g, b)
        
        if max_val == r:
            if r - g < 30:  # Reddish with some green
                return 'orange'
            return 'red'
        elif max_val == g:
            if g - r < 30:  # Greenish with some red
                return 'yellow'
            return 'green'
        else:  # max_val == b
            return 'blue'
    
    def determine_result(self, color_name, rgb_values, test_type):
        """Determine test result based on color and test type"""
        reference = self.color_references.get(test_type, {})
        
        # Print for debugging
        print(f"Detected color name: {color_name}")
        print(f"RGB values: {rgb_values}")
        print(f"Available references: {list(reference.keys())}")
        
        # SPECIAL HANDLING FOR HARDNESS TEST
        if test_type == 'hardness':
            # Extract RGB values
            r, g, b = rgb_values
            
            # Calculate color characteristics
            total = r + g + b
            if total > 0:
                r_ratio = r / total
                g_ratio = g / total
                b_ratio = b / total
                
                print(f"HARDNESS TEST - RGB Ratios - R:{r_ratio:.3f}, G:{g_ratio:.3f}, B:{b_ratio:.3f}")
                
                # Light Blue / Very Soft (0-50 mg/L)
                if b > 200 and g > 150 and r < 150:
                    if b > 230 and g > 180:
                        return {
                            'color': 'Light Blue',
                            'value': 25,
                            'level': 'Very Soft',
                            'unit': 'mg/L'
                        }
                    else:
                        return {
                            'color': 'Blue',
                            'value': 50,
                            'level': 'Very Soft',
                            'unit': 'mg/L'
                        }
                
                # Blue / Soft (50-100 mg/L)
                if b > 180 and g > 120 and g < 180 and r < 120:
                    return {
                        'color': 'Blue',
                        'value': 75,
                        'level': 'Soft',
                        'unit': 'mg/L'
                    }
                
                # Blue-Green / Slightly Hard (100-150 mg/L)
                if b > 150 and g > 150 and r < 150:
                    if g > b:
                        return {
                            'color': 'Blue-Green',
                            'value': 125,
                            'level': 'Slightly Hard',
                            'unit': 'mg/L'
                        }
                    else:
                        return {
                            'color': 'Blue',
                            'value': 100,
                            'level': 'Soft',
                            'unit': 'mg/L'
                        }
                
                # Green / Moderately Hard (150-200 mg/L)
                if g > 180 and r < 150 and b < 150:
                    if g > 200:
                        return {
                            'color': 'Green',
                            'value': 175,
                            'level': 'Moderately Hard',
                            'unit': 'mg/L'
                        }
                    else:
                        return {
                            'color': 'Blue-Green',
                            'value': 150,
                            'level': 'Slightly Hard',
                            'unit': 'mg/L'
                        }
                
                # Green-Yellow / Hard (200-250 mg/L)
                if g > 180 and r > 150 and r < 200 and b < 120:
                    return {
                        'color': 'Green-Yellow',
                        'value': 225,
                        'level': 'Hard',
                        'unit': 'mg/L'
                    }
                
                # Yellow / Very Hard (250-300 mg/L)
                if r > 200 and g > 180 and b < 100:
                    return {
                        'color': 'Yellow',
                        'value': 275,
                        'level': 'Very Hard',
                        'unit': 'mg/L'
                    }
                
                # Orange / Extremely Hard (300-400 mg/L)
                if r > 210 and g > 120 and g < 180 and b < 80:
                    return {
                        'color': 'Orange',
                        'value': 350,
                        'level': 'Extremely Hard',
                        'unit': 'mg/L'
                    }
                
                # Red / Severe (400+ mg/L)
                if r > 200 and g < 120 and b < 80:
                    return {
                        'color': 'Red',
                        'value': 450,
                        'level': 'Severe',
                        'unit': 'mg/L'
                    }
        
        # SPECIAL HANDLING FOR IRON TEST
        if test_type == 'iron':
            # Extract RGB values
            r, g, b = rgb_values
            
            # Calculate color intensity and ratios
            total = r + g + b
            if total > 0:
                r_ratio = r / total
                g_ratio = g / total
                b_ratio = b / total
                
                # Log ratios for debugging
                print(f"IRON TEST - RGB Ratios - R:{r_ratio:.3f}, G:{g_ratio:.3f}, B:{b_ratio:.3f}")
                
                # Very Light Yellow / None (0.02 mg/L) - Almost white
                if r > 245 and g > 245 and b > 240:
                    return {
                        'color': 'Very Light Yellow',
                        'value': 0.02,
                        'level': 'None',
                        'unit': 'mg/L'
                    }
                
                # Light Yellow / None (0.05 mg/L)
                if r > 240 and g > 240 and b > 220:
                    return {
                        'color': 'Light Yellow',
                        'value': 0.05,
                        'level': 'None',
                        'unit': 'mg/L'
                    }
                
                # Pale Yellow / Trace (0.1 mg/L)
                if r > 230 and g > 230 and b > 200:
                    return {
                        'color': 'Pale Yellow',
                        'value': 0.1,
                        'level': 'Trace',
                        'unit': 'mg/L'
                    }
                
                # Yellow / Low (0.2 mg/L)
                if r > 220 and g > 210 and b < 180:
                    if b < 150:
                        return {
                            'color': 'Yellow',
                            'value': 0.2,
                            'level': 'Low',
                            'unit': 'mg/L'
                        }
                    else:
                        return {
                            'color': 'Pale Yellow',
                            'value': 0.1,
                            'level': 'Trace',
                            'unit': 'mg/L'
                        }
                
                # Golden Yellow / Low-Moderate (0.4 mg/L)
                if r > 220 and g > 190 and b < 120:
                    if r > 235 and g > 210:
                        return {
                            'color': 'Golden Yellow',
                            'value': 0.4,
                            'level': 'Low-Moderate',
                            'unit': 'mg/L'
                        }
                    else:
                        return {
                            'color': 'Yellow',
                            'value': 0.2,
                            'level': 'Low',
                            'unit': 'mg/L'
                        }
                
                # Orange-Yellow / Moderate (0.6 mg/L)
                if r > 210 and g > 160 and g < 200 and b < 90:
                    return {
                        'color': 'Orange-Yellow',
                        'value': 0.6,
                        'level': 'Moderate',
                        'unit': 'mg/L'
                    }
                
                # Orange / Moderate (1.0 mg/L)
                if r > 210 and g > 120 and g < 170 and b < 70:
                    return {
                        'color': 'Orange',
                        'value': 1.0,
                        'level': 'Moderate',
                        'unit': 'mg/L'
                    }
                
                # Deep Orange / Moderate-High (1.5 mg/L)
                if r > 200 and g > 90 and g < 140 and b < 50:
                    return {
                        'color': 'Deep Orange',
                        'value': 1.5,
                        'level': 'Moderate-High',
                        'unit': 'mg/L'
                    }
                
                # Red-Orange / High (2.0 mg/L)
                if r > 190 and g > 60 and g < 110 and b < 40:
                    return {
                        'color': 'Red-Orange',
                        'value': 2.0,
                        'level': 'High',
                        'unit': 'mg/L'
                    }
                
                # Red-Brown / Very High (3.0 mg/L)
                if r > 160 and r < 210 and g > 40 and g < 80 and b < 35:
                    return {
                        'color': 'Red-Brown',
                        'value': 3.0,
                        'level': 'Very High',
                        'unit': 'mg/L'
                    }
                
                # Brown / Severe (5.0 mg/L)
                if r > 120 and r < 180 and g > 30 and g < 70 and b < 30:
                    return {
                        'color': 'Brown',
                        'value': 5.0,
                        'level': 'Severe',
                        'unit': 'mg/L'
                    }
                
                # Dark Brown / Severe (7.0 mg/L)
                if r > 80 and r < 150 and g > 20 and g < 60 and b < 25:
                    return {
                        'color': 'Dark Brown',
                        'value': 7.0,
                        'level': 'Severe',
                        'unit': 'mg/L'
                    }
        
        # For other test types, use the existing matching logic
        # Direct match (case-insensitive)
        for key in reference.keys():
            if color_name.lower() in key.lower() or key.lower() in color_name.lower():
                ref_data = reference[key]
                print(f"Matched to: {key}")
                return {
                    'color': ref_data['color'],
                    'value': ref_data['value'],
                    'level': ref_data['level'],
                    'unit': ref_data['unit']
                }
        
        # If no direct match, try to find closest match
        print("No direct match, finding closest...")
        closest_match = self.find_closest_color(rgb_values, reference, test_type)
        if closest_match:
            return closest_match
        
        # Default fallback based on test type
        print("Using default fallback")
        if test_type == 'ph':
            return {
                'color': 'Unknown',
                'value': 7.0,
                'level': 'Neutral',
                'unit': 'pH'
            }
        elif test_type == 'chlorine':
            return {
                'color': 'Yellow',
                'value': 2.5,
                'level': 'Moderate',
                'unit': 'mg/L'
            }
        elif test_type == 'iron':
            return {
                'color': 'Light Yellow',
                'value': 0.05,
                'level': 'None',
                'unit': 'mg/L'
            }
        elif test_type == 'hardness':
            return {
                'color': 'Blue',
                'value': 100,
                'level': 'Soft',
                'unit': 'mg/L'
            }
        else:
            return {
                'color': 'Unknown',
                'value': 0.5,
                'level': 'Low',
                'unit': 'mg/L'
            }
    
    def find_closest_color(self, rgb_values, reference, test_type=None):
        """Find closest color in reference with improved matching"""
        min_distance = float('inf')
        best_match = None
        
        # Enhanced color approximations with more precise values
        color_approximations = {
            # Hardness test colors
            'light_blue': [200, 220, 250],
            'blue': [100, 150, 230],
            'blue_green': [80, 180, 180],
            'green': [50, 200, 120],
            'green_yellow': [150, 200, 80],
            'yellow': [230, 230, 100],
            'orange': [240, 160, 60],
            'red': [230, 80, 70],
            
            # Iron test colors
            'very_light_yellow': [250, 250, 240],
            'light_yellow': [245, 245, 220],
            'pale_yellow': [240, 240, 200],
            'yellow': [235, 235, 150],
            'golden_yellow': [240, 220, 100],
            'orange_yellow': [235, 190, 70],
            'orange': [230, 160, 50],
            'deep_orange': [220, 130, 40],
            'red_orange': [210, 90, 35],
            'red_brown': [180, 70, 40],
            'brown': [150, 60, 35],
            'dark_brown': [120, 50, 30],
            
            # Other test colors
            'red': [255, 50, 50],
            'green': [50, 255, 50],
            'blue': [50, 50, 255],
            'purple': [180, 50, 180],
            'pink': [255, 180, 180],
            'light_pink': [255, 200, 200],
            'medium_pink': [255, 150, 150],
            'dark_pink': [230, 100, 130],
            'purple_pink': [220, 150, 220]
        }
        
        for key, ref_data in reference.items():
            best_distance_for_key = float('inf')
            best_match_for_key = None
            
            for ref_key, approx_rgb in color_approximations.items():
                # Check if this approximation matches the reference key
                if ref_key in key.lower() or key.lower() in ref_key:
                    # Calculate different distance metrics
                    
                    # Standard Euclidean distance
                    euclidean_dist = np.sqrt(
                        (rgb_values[0] - approx_rgb[0]) ** 2 +
                        (rgb_values[1] - approx_rgb[1]) ** 2 +
                        (rgb_values[2] - approx_rgb[2]) ** 2
                    )
                    
                    # Weighted distance for better color perception
                    if test_type == 'iron' and ('yellow' in key.lower() or 'orange' in key.lower()):
                        weighted_dist = np.sqrt(
                            1.5 * (rgb_values[0] - approx_rgb[0]) ** 2 +
                            1.5 * (rgb_values[1] - approx_rgb[1]) ** 2 +
                            0.5 * (rgb_values[2] - approx_rgb[2]) ** 2
                        )
                    elif test_type == 'hardness' and ('blue' in key.lower() or 'green' in key.lower()):
                        weighted_dist = np.sqrt(
                            1.2 * (rgb_values[0] - approx_rgb[0]) ** 2 +
                            1.2 * (rgb_values[1] - approx_rgb[1]) ** 2 +
                            1.2 * (rgb_values[2] - approx_rgb[2]) ** 2
                        )
                    else:
                        weighted_dist = euclidean_dist
                    
                    # Use the minimum of both metrics
                    distance = min(euclidean_dist, weighted_dist)
                    
                    if distance < best_distance_for_key:
                        best_distance_for_key = distance
                        best_match_for_key = {
                            'color': ref_data['color'],
                            'value': ref_data['value'],
                            'level': ref_data['level'],
                            'unit': ref_data['unit'],
                            'distance': distance
                        }
            
            if best_match_for_key and best_distance_for_key < min_distance:
                min_distance = best_distance_for_key
                best_match = best_match_for_key
                print(f"Found match: {best_match['color']} with distance {min_distance:.2f}")
        
        return best_match

web_app = WaterTestWebApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        before_image = data.get('before_image')
        after_image = data.get('after_image')
        test_type = data.get('test_type', 'ph')
        
        if not before_image or not after_image:
            return jsonify({'error': 'Both images required'}), 400
        
        # Process images
        before_processed, before_raw = web_app.process_image(before_image)
        after_processed, after_raw = web_app.process_image(after_image)
        
        if before_processed is None or after_processed is None:
            return jsonify({'error': 'Could not process images'}), 400
        
        # Analyze colors
        before_result = web_app.analyze_color(before_processed, test_type)
        after_result = web_app.analyze_color(after_processed, test_type)
        
        # Calculate change
        color_change = np.linalg.norm(
            np.array(before_result['rgb']) - np.array(after_result['rgb'])
        )
        
        # Compare images
        before_region = web_app.image_processor.extract_test_region(
            before_processed, 
            web_app.image_processor.detect_test_strip(before_processed)
        )
        after_region = web_app.image_processor.extract_test_region(
            after_processed, 
            web_app.image_processor.detect_test_strip(after_processed)
        )
        
        comparison = web_app.image_processor.compare_images(before_region, after_region)
        change_analysis = web_app.image_processor.detect_color_change_region(before_region, after_region)
        
        # Generate response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'test_type': test_type,
            'before': before_result,
            'after': after_result,
            'comparison': {
                'color_change': float(color_change),
                'mse': float(comparison['mse']),
                'histogram_similarity': float(comparison['histogram_difference']),
                'change_percentage': float(change_analysis['change_percentage'])
            },
            'result': after_result['result'],
            'explanation': get_explanation(test_type, after_result['color_name'], after_result['result']['value']),
            'recommendation': get_recommendation(test_type, after_result['result']['level']),
            'ml_analysis': after_result.get('ml_analysis')
        }
        
        # Store result
        web_app.test_results.append(response)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get test history"""
    return jsonify(web_app.test_results[-10:])  # Last 10 results

@app.route('/export/<int:index>')
def export_result(index):
    """Export specific result as JSON"""
    if 0 <= index < len(web_app.test_results):
        return jsonify(web_app.test_results[index])
    return jsonify({'error': 'Result not found'}), 404

# ==================== DOWNLOAD ENDPOINTS ====================

@app.route('/download/<int:index>/<format>')
def download_result(index, format):
    """Download specific test result in specified format"""
    try:
        if index < 0 or index >= len(web_app.test_results):
            return jsonify({'error': 'Result not found'}), 404
        
        result = web_app.test_results[index]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            # Download as JSON
            filename = f"water_test_{result['test_type']}_{timestamp}.json"
            result_json = json.dumps(result, indent=4, default=str)
            
            return Response(
                result_json,
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        
        elif format == 'txt':
            # Download as text file
            filename = f"water_test_{result['test_type']}_{timestamp}.txt"
            content = format_result_as_text(result)
            
            return Response(
                content,
                mimetype='text/plain',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        
        elif format == 'csv':
            # Download as CSV
            filename = f"water_test_{result['test_type']}_{timestamp}.csv"
            content = format_result_as_csv(result)
            
            return Response(
                content,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-all/<format>')
def download_all_results(format):
    """Download all test results in specified format"""
    try:
        if not web_app.test_results:
            return jsonify({'error': 'No results available'}), 404
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            # Download all as JSON array
            filename = f"all_water_tests_{timestamp}.json"
            result_json = json.dumps(web_app.test_results, indent=4, default=str)
            
            return Response(
                result_json,
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        
        elif format == 'csv':
            # Download all as CSV
            filename = f"all_water_tests_{timestamp}.csv"
            content = format_all_as_csv(web_app.test_results)
            
            return Response(
                content,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== FORMATTING FUNCTIONS ====================

def format_result_as_text(result):
    """Format a single result as text"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"WATER QUALITY TEST REPORT")
    lines.append("=" * 70)
    lines.append(f"Test Type: {result['test_type'].upper()}")
    lines.append(f"Date/Time: {result['timestamp']}")
    lines.append(f"Detected Color: {result['after']['color_name']}")
    lines.append(f"RGB Values: {result['after']['rgb']}")
    
    if result.get('ml_analysis'):
        lines.append(f"ML Analysis: {result['ml_analysis']['color_name']} "
                    f"(confidence: {result['ml_analysis']['confidence']*100:.1f}%)")
    
    lines.append("\n" + "-" * 70)
    lines.append("TEST RESULTS:")
    lines.append(f"  • Color: {result['result']['color']}")
    lines.append(f"  • Concentration: {result['result']['value']} {result['result']['unit']}")
    lines.append(f"  • Level: {result['result']['level']}")
    
    lines.append("\n" + "-" * 70)
    lines.append("SCIENTIFIC EXPLANATION:")
    lines.append(f"  {result['explanation']}")
    
    lines.append("\n" + "-" * 70)
    lines.append("SAFETY RECOMMENDATION:")
    lines.append(f"  {result['recommendation']}")
    
    if 'comparison' in result:
        lines.append("\n" + "-" * 70)
        lines.append("COLOR ANALYSIS:")
        lines.append(f"  • Color Change: {result['comparison']['color_change']:.2f}")
        lines.append(f"  • Change Percentage: {result['comparison']['change_percentage']:.1f}%")
        lines.append(f"  • MSE: {result['comparison']['mse']:.2f}")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)

def format_result_as_csv(result):
    """Format a single result as CSV"""
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Field', 'Value'])
    
    # Write data
    writer.writerow(['Test Type', result['test_type'].upper()])
    writer.writerow(['Timestamp', result['timestamp']])
    writer.writerow(['Detected Color', result['after']['color_name']])
    writer.writerow(['RGB', str(result['after']['rgb'])])
    writer.writerow(['Concentration', f"{result['result']['value']} {result['result']['unit']}"])
    writer.writerow(['Level', result['result']['level']])
    writer.writerow(['Explanation', result['explanation']])
    writer.writerow(['Recommendation', result['recommendation']])
    
    if 'comparison' in result:
        writer.writerow(['Color Change', result['comparison']['color_change']])
        writer.writerow(['Change Percentage', result['comparison']['change_percentage']])
    
    return output.getvalue()

def format_all_as_csv(results):
    """Format all results as CSV"""
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Index', 'Timestamp', 'Test Type', 'Color', 'RGB', 
                    'Concentration', 'Unit', 'Level', 'Explanation', 'Recommendation'])
    
    # Write data for each result
    for i, result in enumerate(results, 1):
        writer.writerow([
            i,
            result['timestamp'],
            result['test_type'].upper(),
            result['after']['color_name'],
            str(result['after']['rgb']),
            result['result']['value'],
            result['result']['unit'],
            result['result']['level'],
            result['explanation'],
            result['recommendation']
        ])
    
    return output.getvalue()

# ==================== EXISTING FUNCTIONS ====================

def get_explanation(test_type, color, value):
    explanations = {
        'ph': f"The color changed to {color} because pH indicator molecules gain or lose protons (H⁺) depending on the solution's acidity. The measured pH value of {value} indicates the concentration of hydrogen ions.",
        'iron': f"The appearance of a {color} color indicates iron presence. The concentration of {value} mg/L represents the amount of iron in the sample.",
        'hardness': f"The color changed to {color} indicating water hardness. The concentration of {value} mg/L as CaCO₃ represents the total hardness level. Hardness is caused by dissolved calcium and magnesium ions.",
        'chlorine': f"The appearance of a {color} color indicates chlorine presence through DPD reaction. The concentration of {value} mg/L represents free chlorine levels."
    }
    return explanations.get(test_type, "Color change indicates presence of target analyte.")

def get_recommendation(test_type, level):
    recommendations = {
        'ph': {
            'Strong Acid': '⚠️ CRITICAL: Immediate pH adjustment required. Do not consume.',
            'Acidic': '⚠️ Consider acid-neutralizing filter installation.',
            'Slightly Acidic': '⚠️ Monitor regularly for changes.',
            'Neutral': '✅ Water is safe for consumption.',
            'Slightly Basic': '⚠️ Monitor for scale formation.',
            'Basic': '⚠️ Consider water softening if problems persist.'
        },
        'iron': {
            'None': '✅ No detectable iron. Water is safe.',
            'Trace': '✅ Trace amounts. Safe for consumption.',
            'Low': '✅ Low iron levels. Safe for consumption.',
            'Low-Moderate': '⚠️ Low to moderate iron. Monitor regularly.',
            'Moderate': '⚠️ Moderate iron. Consider filtration if staining occurs.',
            'Moderate-High': '⚠️ Moderate to high iron. Filtration recommended.',
            'High': '⚠️ High iron. Iron filtration recommended.',
            'Very High': '⚠️ Very high iron. Professional treatment required.',
            'Severe': '⚠️ CRITICAL: Severe iron contamination. Do not consume.'
        },
        'hardness': {
            'Very Soft': '✅ Very soft water. May be corrosive to pipes. Consider pH adjustment.',
            'Soft': '✅ Soft water. Good for most household uses.',
            'Slightly Hard': '✅ Slightly hard water. Acceptable for most uses.',
            'Moderately Hard': '⚠️ Moderately hard water. May cause scale buildup.',
            'Hard': '⚠️ Hard water. Scale formation likely. Consider water softener.',
            'Very Hard': '⚠️ Very hard water. Significant scale buildup. Water softener recommended.',
            'Extremely Hard': '⚠️ Extremely hard water. Professional water softening required.',
            'Severe': '⚠️ CRITICAL: Severe hardness. Treatment mandatory.'
        },
        'chlorine': {
            'Very Low': '⚠️ Inadequate disinfection. Increase chlorine dosage.',
            'Low': '⚠️ Minimal disinfection capability. Monitor closely.',
            'Low-Moderate': '✅ Adequate disinfection for most applications.',
            'Moderate': '✅ Optimal range for disinfection.',
            'Moderate-High': '⚠️ May cause taste/odor but safe.',
            'High': '⚠️ Strong taste/odor. Consider reducing dosage.',
            'Very High': '⚠️ CRITICAL: Unsafe levels. Reduce immediately.'
        }
    }
    return recommendations.get(test_type, {}).get(level, 'Consult water specialist.')

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the ML model"""
    try:
        web_app.ml_classifier.train()
        web_app.ml_classifier.save_model("color_classifier.pkl")
        return jsonify({'success': True, 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)