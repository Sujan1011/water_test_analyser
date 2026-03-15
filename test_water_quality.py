#!/usr/bin/env python3
"""
Water Quality Test Examiner
This script allows you to test water quality by providing RGB values or image files
and get the analysis results for different test types.
"""

import sys
import os
import json
import numpy as np
import cv2
from datetime import datetime

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from image_processor import ImageProcessor
from ml_classifier import ColorClassifier

class WaterTestExaminer:
    def __init__(self):
        """Initialize the water test examiner"""
        self.image_processor = ImageProcessor()
        self.ml_classifier = ColorClassifier()
        
        # Try to load pre-trained model
        self.initialize_classifier()
        
        # Color reference database
        self.color_references = self.load_color_references()
        
        print("=" * 60)
        print("WATER QUALITY TEST EXAMINER")
        print("=" * 60)
        
    def initialize_classifier(self):
        """Initialize the ML classifier"""
        try:
            if os.path.exists("color_classifier.pkl"):
                self.ml_classifier.load_model("color_classifier.pkl")
                print("✅ Loaded pre-trained classifier")
            else:
                print("⚠️  No pre-trained model found. Training new classifier...")
                self.ml_classifier.train()
                self.ml_classifier.save_model("color_classifier.pkl")
                print("✅ Classifier trained and saved")
        except Exception as e:
            print(f"⚠️  Classifier initialization warning: {e}")
            print("Using rule-based color detection fallback")
    
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
                'light_yellow': {'range': (0, 0.1), 'color': 'Light Yellow', 'level': 'None', 'value': 0.05, 'unit': 'mg/L'},
                'yellow': {'range': (0.1, 0.3), 'color': 'Yellow', 'level': 'Low', 'value': 0.2, 'unit': 'mg/L'},
                'orange': {'range': (0.3, 1.0), 'color': 'Orange', 'level': 'Medium', 'value': 0.6, 'unit': 'mg/L'},
                'red_orange': {'range': (1.0, 3.0), 'color': 'Red-Orange', 'level': 'High', 'value': 2.0, 'unit': 'mg/L'},
                'red_brown': {'range': (3.0, 5.0), 'color': 'Red-Brown', 'level': 'Very High', 'value': 4.0, 'unit': 'mg/L'},
                'brown': {'range': (5.0, 10), 'color': 'Brown', 'level': 'Severe', 'value': 7.0, 'unit': 'mg/L'}
            },
            'nitrate': {
                'light_pink': {'range': (0, 1), 'color': 'Light Pink', 'level': 'None', 'value': 0.5, 'unit': 'mg/L'},
                'pink': {'range': (1, 5), 'color': 'Pink', 'level': 'Low', 'value': 3.0, 'unit': 'mg/L'},
                'medium_pink': {'range': (5, 10), 'color': 'Medium Pink', 'level': 'Medium', 'value': 7.5, 'unit': 'mg/L'},
                'dark_pink': {'range': (10, 20), 'color': 'Dark Pink', 'level': 'High', 'value': 15.0, 'unit': 'mg/L'},
                'purple_pink': {'range': (20, 50), 'color': 'Purple-Pink', 'level': 'Very High', 'value': 35.0, 'unit': 'mg/L'},
                'purple': {'range': (50, 100), 'color': 'Purple', 'level': 'Severe', 'value': 75.0, 'unit': 'mg/L'}
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
    
    def rgb_to_color_name(self, rgb):
        """Convert RGB to color name with enhanced detection"""
        r, g, b = rgb
        
        # Enhanced color classification
        if r > 240 and g > 240 and b > 240:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > 200 and g > 200 and b < 150:
            if r > 240 and g > 240:
                return 'very_light_yellow'
            elif r > 220 and g > 220:
                return 'light_yellow'
            elif r > 200 and g > 200:
                return 'pale_yellow'
            else:
                return 'yellow'
        elif r > 200 and g > 180 and b < 100:
            return 'golden_yellow'
        elif r > 180 and g > 180 and b < 80:
            return 'deep_yellow'
        elif r > 200 and g > 150 and b < 100:
            return 'orange_yellow'
        elif r > 200 and g < 100 and b < 100:
            return 'red'
        elif r > 200 and g > 150 and b < 100:
            if r > 220 and g < 180:
                return 'orange'
            else:
                return 'red_orange'
        elif r < 150 and g > 150 and b < 150:
            return 'green'
        elif r < 150 and g < 150 and b > 150:
            return 'blue'
        elif r > 150 and g < 150 and b > 150:
            return 'purple'
        elif r > 200 and g < 200 and b > 200:
            return 'purple_pink'
        elif r > 200 and g < 180 and b > 150:
            return 'dark_pink'
        elif r > 200 and g < 160 and b > 100:
            return 'medium_pink'
        elif r > 180 and g < 140 and b > 80:
            return 'pink'
        elif r > 160 and g < 120 and b > 60:
            return 'light_pink'
        else:
            # Default to closest by max value
            max_val = max(r, g, b)
            if max_val == r:
                return 'red'
            elif max_val == g:
                return 'green'
            else:
                return 'blue'
    
    def match_color_to_reference(self, color_name, rgb_values, test_type):
        """Match detected color to reference values"""
        reference = self.color_references.get(test_type, {})
        
        # Direct match
        if color_name in reference:
            ref_data = reference[color_name]
            return {
                'color': ref_data['color'],
                'level': ref_data['level'],
                'value': ref_data['value'],
                'unit': ref_data['unit']
            }
        
        # Try to find closest match
        closest_match = self.find_closest_color(rgb_values, reference)
        if closest_match:
            return closest_match
        
        # Default fallback
        default_key = list(reference.keys())[len(reference)//2] if reference else None
        if default_key:
            ref_data = reference[default_key]
            return {
                'color': ref_data['color'],
                'level': ref_data['level'],
                'value': ref_data['value'],
                'unit': ref_data['unit']
            }
        else:
            return {
                'color': 'Unknown',
                'level': 'Unknown',
                'value': 0,
                'unit': 'N/A'
            }
    
    def find_closest_color(self, rgb_values, reference):
        """Find closest color in reference based on RGB distance"""
        min_distance = float('inf')
        best_match = None
        
        color_approximations = {
            'red': [255, 0, 0],
            'orange': [255, 165, 0],
            'yellow': [255, 255, 0],
            'green': [0, 255, 0],
            'blue': [0, 0, 255],
            'purple': [128, 0, 128],
            'pink': [255, 192, 203],
            'brown': [139, 69, 19],
            'light_yellow': [255, 255, 224],
            'golden_yellow': [255, 215, 0],
            'deep_yellow': [255, 204, 0],
            'orange_yellow': [255, 200, 0],
            'light_pink': [255, 182, 193],
            'medium_pink': [255, 105, 180],
            'dark_pink': [255, 20, 147],
            'purple_pink': [221, 160, 221],
            'red_orange': [255, 69, 0],
            'red_brown': [165, 42, 42]
        }
        
        for key, ref_data in reference.items():
            if key in color_approximations:
                approx_rgb = color_approximations[key]
                distance = np.linalg.norm(np.array(rgb_values) - np.array(approx_rgb))
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = {
                        'color': ref_data['color'],
                        'level': ref_data['level'],
                        'value': ref_data['value'],
                        'unit': ref_data['unit']
                    }
        
        return best_match
    
    def get_scientific_explanation(self, test_type, color, value):
        """Get scientific explanation for the result"""
        explanations = {
            'ph': f"The color {color} indicates a pH value of {value}. pH indicator molecules gain or lose protons (H⁺) depending on the solution's acidity.",
            'iron': f"The {color} color indicates iron presence at {value} mg/L. Iron ions form a coordination complex with the reagent.",
            'nitrate': f"The {color} color shows nitrate presence at {value} mg/L through the Griess reaction.",
            'chlorine': f"The {color} color indicates chlorine presence at {value} mg/L through DPD reaction."
        }
        return explanations.get(test_type, "Color change indicates presence of target analyte.")
    
    def get_safety_recommendation(self, test_type, level, value):
        """Get safety recommendations based on results"""
        recommendations = {
            'ph': {
                'Strong Acid': "⚠️ CRITICAL: Water is strongly acidic. Do not consume.",
                'Acidic': "⚠️ Acidic water. Consider treatment.",
                'Slightly Acidic': "⚠️ Slightly acidic. Monitor regularly.",
                'Neutral': "✅ Water is safe for consumption.",
                'Slightly Basic': "⚠️ Slightly basic. Monitor for scale.",
                'Basic': "⚠️ Basic water. Consider treatment."
            },
            'iron': {
                'None': "✅ No detectable iron. Water is safe.",
                'Low': "✅ Low iron levels. Safe for consumption.",
                'Medium': "⚠️ Moderate iron. Consider filtration.",
                'High': "⚠️ High iron. Treatment recommended.",
                'Very High': "⚠️ Very high iron. Treatment required.",
                'Severe': "⚠️ CRITICAL: Severe iron contamination."
            },
            'nitrate': {
                'None': "✅ No detectable nitrate. Safe for all uses.",
                'Low': "✅ Low nitrate. Safe for consumption.",
                'Medium': "⚠️ Moderate nitrate. Caution for infants.",
                'High': "⚠️ High nitrate. Unsafe for infants.",
                'Very High': "⚠️ CRITICAL: Very high nitrate.",
                'Severe': "⚠️ DANGEROUS: Water is toxic."
            },
            'chlorine': {
                'Very Low': "⚠️ Very low chlorine. Inadequate disinfection.",
                'Low': "⚠️ Low chlorine. Monitor closely.",
                'Low-Moderate': "✅ Acceptable chlorine levels.",
                'Moderate': "✅ Optimal chlorine range.",
                'Moderate-High': "⚠️ Elevated chlorine. Safe but noticeable.",
                'High': "⚠️ High chlorine. Consider reducing.",
                'Very High': "⚠️ CRITICAL: Unsafe chlorine levels."
            }
        }
        return recommendations.get(test_type, {}).get(level, "Consult water specialist.")
    
    def test_from_rgb(self, rgb_values, test_type='ph'):
        """
        Test water quality from RGB values
        
        Args:
            rgb_values: List of [R, G, B] values (0-255)
            test_type: Type of test ('ph', 'iron', 'nitrate', 'chlorine')
        
        Returns:
            Dictionary with test results
        """
        print(f"\n🔍 Testing {test_type.upper()} with RGB: {rgb_values}")
        
        # Get color name
        color_name = self.rgb_to_color_name(rgb_values)
        print(f"📊 Detected color: {color_name}")
        
        # Get ML prediction if available
        ml_result = None
        try:
            if self.ml_classifier.is_trained:
                ml_result = self.ml_classifier.predict_color(rgb_values)
                print(f"🤖 ML prediction: {ml_result['color_name']} (confidence: {ml_result['confidence']*100:.1f}%)")
        except:
            pass
        
        # Match to reference
        result = self.match_color_to_reference(color_name, rgb_values, test_type)
        
        # Add explanation and recommendation
        result['explanation'] = self.get_scientific_explanation(test_type, result['color'], result['value'])
        result['recommendation'] = self.get_safety_recommendation(test_type, result['level'], result['value'])
        
        if ml_result:
            result['ml_confidence'] = ml_result['confidence']
            result['ml_color'] = ml_result['color_name']
        
        return {
            'test_type': test_type,
            'rgb': rgb_values,
            'color_name': color_name,
            'result': result,
            'ml_analysis': ml_result
        }
    
    def test_from_image(self, image_path, test_type='ph'):
        """
        Test water quality from an image file
        
        Args:
            image_path: Path to image file
            test_type: Type of test ('ph', 'iron', 'nitrate', 'chlorine')
        
        Returns:
            Dictionary with test results
        """
        print(f"\n🔍 Testing {test_type.upper()} from image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return None
        
        # Process image
        processed = self.image_processor.preprocess_image(image_path)
        if processed is None:
            print("❌ Error: Could not process image")
            return None
        
        # Extract test region
        strip_region = self.image_processor.detect_test_strip(processed)
        test_region = self.image_processor.extract_test_region(processed, strip_region)
        
        # Analyze color distribution
        color_distribution = self.image_processor.analyze_color_distribution(test_region)
        
        # Get dominant color
        dominant_color = color_distribution[0]['rgb'] if color_distribution else [128, 128, 128]
        
        print(f"🎨 Dominant color RGB: {dominant_color}")
        print(f"📊 Color distribution: {len(color_distribution)} colors detected")
        
        # Test from RGB
        return self.test_from_rgb(dominant_color, test_type)
    
    def print_result(self, result):
        """Print formatted test results"""
        if not result:
            print("❌ No results to display")
            return
        
        print("\n" + "=" * 60)
        print(f"📋 TEST RESULTS - {result['test_type'].upper()}")
        print("=" * 60)
        print(f"🎨 Detected Color: {result['color_name']}")
        print(f"🎯 RGB Values: {result['rgb']}")
        
        if 'ml_analysis' in result and result['ml_analysis']:
            print(f"🤖 ML Analysis: {result['ml_analysis']['color_name']} (confidence: {result['ml_analysis']['confidence']*100:.1f}%)")
        
        print("\n" + "-" * 60)
        print("📊 ANALYSIS:")
        print(f"   • Color: {result['result']['color']}")
        print(f"   • Concentration: {result['result']['value']} {result['result']['unit']}")
        print(f"   • Level: {result['result']['level']}")
        
        print("\n" + "-" * 60)
        print("🔬 SCIENTIFIC EXPLANATION:")
        print(f"   {result['result']['explanation']}")
        
        print("\n" + "-" * 60)
        print("⚠️  SAFETY RECOMMENDATION:")
        print(f"   {result['result']['recommendation']}")
        print("=" * 60)

def interactive_mode():
    """Run interactive test mode with save functionality"""
    examiner = WaterTestExaminer()
    
    while True:
        print("\n" + "=" * 60)
        print("INTERACTIVE TEST MENU")
        print("=" * 60)
        print("1. Test from RGB values")
        print("2. Test from image file")
        print("3. Run all test examples")
        print("4. View saved results")
        print("5. Export all results to CSV")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            # Test from RGB
            print("\n--- Test from RGB Values ---")
            try:
                r = int(input("Enter Red value (0-255): "))
                g = int(input("Enter Green value (0-255): "))
                b = int(input("Enter Blue value (0-255): "))
                
                print("\nSelect test type:")
                print("1. pH Test")
                print("2. Iron Test")
                print("3. Nitrate Test")
                print("4. Chlorine Test")
                
                test_choice = input("Enter choice (1-4): ").strip()
                test_types = {'1': 'ph', '2': 'iron', '3': 'nitrate', '4': 'chlorine'}
                test_type = test_types.get(test_choice, 'ph')
                
                result = examiner.test_from_rgb([r, g, b], test_type)
                examiner.print_result_with_save_option(result)
                
            except ValueError:
                print("❌ Invalid input. Please enter numbers.")
        
        elif choice == '2':
            # Test from image
            print("\n--- Test from Image File ---")
            image_path = input("Enter image file path: ").strip()
            
            print("\nSelect test type:")
            print("1. pH Test")
            print("2. Iron Test")
            print("3. Nitrate Test")
            print("4. Chlorine Test")
            
            test_choice = input("Enter choice (1-4): ").strip()
            test_types = {'1': 'ph', '2': 'iron', '3': 'nitrate', '4': 'chlorine'}
            test_type = test_types.get(test_choice, 'ph')
            
            result = examiner.test_from_image(image_path, test_type)
            if result:
                examiner.print_result_with_save_option(result)
        
        elif choice == '3':
            # Run all examples
            print("\n--- Running All Test Examples ---")
            
            test_examples = [
                ('ph', [255, 0, 0], "Red - Strong Acid"),
                ('ph', [0, 255, 0], "Green - Neutral"),
                ('ph', [0, 0, 255], "Blue - Slightly Basic"),
                ('iron', [255, 255, 200], "Light Yellow - None"),
                ('iron', [255, 165, 0], "Orange - Medium"),
                ('iron', [139, 69, 19], "Brown - Severe"),
                ('nitrate', [255, 192, 203], "Light Pink - None"),
                ('nitrate', [255, 105, 180], "Medium Pink - Medium"),
                ('nitrate', [128, 0, 128], "Purple - Severe"),
                ('chlorine', [255, 255, 220], "Very Light Yellow - Very Low"),
                ('chlorine', [255, 215, 0], "Golden Yellow - Moderate-High"),
                ('chlorine', [255, 140, 0], "Deep Yellow - High")
            ]
            
            all_results = []
            for test_type, rgb, description in test_examples:
                print(f"\n📝 Testing: {description}")
                result = examiner.test_from_rgb(rgb, test_type)
                examiner.print_result(result)
                all_results.append(result)
                input("\nPress Enter to continue...")
            
            # Ask to save all results
            save_all = input("\n💾 Do you want to save all results? (y/n): ").strip().lower()
            if save_all == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"all_test_results_{timestamp}.txt"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    for i, result in enumerate(all_results, 1):
                        f.write(f"\n{'='*70}\n")
                        f.write(f"TEST #{i}\n")
                        f.write(f"{'='*70}\n")
                        f.write(examiner.format_result_for_file(result))
                
                print(f"✅ All results saved to: {filename}")
        
        elif choice == '4':
            # View saved results
            print("\n--- Saved Results ---")
            saved_files = [f for f in os.listdir('.') if f.startswith('water_test_') and f.endswith('.txt')]
            
            if not saved_files:
                print("No saved results found.")
            else:
                print("\nAvailable result files:")
                for i, file in enumerate(saved_files, 1):
                    print(f"{i}. {file}")
                
                file_choice = input("\nEnter file number to view (or 0 to cancel): ").strip()
                if file_choice.isdigit() and 0 < int(file_choice) <= len(saved_files):
                    filename = saved_files[int(file_choice)-1]
                    with open(filename, 'r', encoding='utf-8') as f:
                        print(f"\n{f.read()}")
        
        elif choice == '5':
            # Export all results to CSV
            print("\n--- Export to CSV ---")
            all_results = []
            
            # Collect all saved text results
            saved_files = [f for f in os.listdir('.') if f.startswith('water_test_') and f.endswith('.json')]
            
            if not saved_files:
                print("No JSON results found. Run some tests and save them as JSON first.")
            else:
                csv_filename = f"all_water_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                with open(csv_filename, 'w', encoding='utf-8') as csv_file:
                    # Write CSV header
                    csv_file.write("Timestamp,Test Type,Color,RGB,Concentration,Unit,Level\n")
                    
                    for json_file in saved_files:
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                timestamp = data.get('timestamp', 'N/A')
                                test_type = data.get('test_type', 'N/A')
                                color = data.get('color_name', 'N/A')
                                rgb = str(data.get('rgb', 'N/A'))
                                conc = data.get('result', {}).get('value', 'N/A')
                                unit = data.get('result', {}).get('unit', 'N/A')
                                level = data.get('result', {}).get('level', 'N/A')
                                
                                csv_file.write(f"{timestamp},{test_type},{color},{rgb},{conc},{unit},{level}\n")
                        except:
                            continue
                
                print(f"✅ CSV file created: {csv_filename}")
        
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")
def save_result_to_file(self, result, filename=None):
    """Save test result to a file"""
    if not result:
        print("❌ No results to save")
        return False
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"water_test_{result['test_type']}_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.format_result_for_file(result))
        print(f"✅ Result saved to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def save_result_as_json(self, result, filename=None):
    """Save test result as JSON file"""
    if not result:
        print("❌ No results to save")
        return False
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"water_test_{result['test_type']}_{timestamp}.json"
    
    try:
        # Convert numpy arrays to lists for JSON
        result_json = self.convert_to_serializable(result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)
        print(f"✅ JSON result saved to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        return False

def convert_to_serializable(self, obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: self.convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [self.convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(self.convert_to_serializable(item) for item in obj)
    else:
        return obj

def format_result_for_file(self, result):
    """Format result for text file output"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"WATER QUALITY TEST REPORT")
    lines.append("=" * 70)
    lines.append(f"Test Type: {result['test_type'].upper()}")
    lines.append(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Detected Color: {result['color_name']}")
    lines.append(f"RGB Values: {result['rgb']}")
    
    if 'ml_analysis' in result and result['ml_analysis']:
        lines.append(f"ML Analysis: {result['ml_analysis']['color_name']} "
                    f"(confidence: {result['ml_analysis']['confidence']*100:.1f}%)")
    
    lines.append("\n" + "-" * 70)
    lines.append("TEST RESULTS:")
    lines.append(f"  • Color: {result['result']['color']}")
    lines.append(f"  • Concentration: {result['result']['value']} {result['result']['unit']}")
    lines.append(f"  • Level: {result['result']['level']}")
    
    lines.append("\n" + "-" * 70)
    lines.append("SCIENTIFIC EXPLANATION:")
    lines.append(f"  {result['result']['explanation']}")
    
    lines.append("\n" + "-" * 70)
    lines.append("SAFETY RECOMMENDATION:")
    lines.append(f"  {result['result']['recommendation']}")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)

def print_result_with_save_option(self, result):
    """Print results and ask if user wants to save"""
    self.print_result(result)
    
    if result:
        print("\n" + "-" * 70)
        choice = input("💾 Do you want to save this result? (y/n): ").strip().lower()
        
        if choice == 'y':
            print("\nSave as:")
            print("1. Text file (.txt)")
            print("2. JSON file (.json)")
            print("3. Both")
            
            format_choice = input("Enter choice (1-3): ").strip()
            
            if format_choice == '1':
                self.save_result_to_file(result)
            elif format_choice == '2':
                self.save_result_as_json(result)
            elif format_choice == '3':
                self.save_result_to_file(result)
                self.save_result_as_json(result)
            else:
                print("❌ Invalid choice")
def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Water Quality Test Examiner')
    parser.add_argument('--rgb', nargs=3, type=int, metavar=('R', 'G', 'B'),
                        help='RGB values to test (e.g., --rgb 255 0 0)')
    parser.add_argument('--image', type=str, help='Image file to test')
    parser.add_argument('--test', choices=['ph', 'iron', 'nitrate', 'chlorine'],
                        default='ph', help='Type of test to perform')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save result to file')
    parser.add_argument('--format', choices=['txt', 'json', 'both'],
                        default='txt', help='Output format for saved results')
    
    args = parser.parse_args()
    
    examiner = WaterTestExaminer()
    
    if args.interactive:
        interactive_mode()
    elif args.rgb:
        # Test from RGB command line
        result = examiner.test_from_rgb(args.rgb, args.test)
        examiner.print_result(result)
        
        if args.save:
            if args.format == 'txt' or args.format == 'both':
                examiner.save_result_to_file(result)
            if args.format == 'json' or args.format == 'both':
                examiner.save_result_as_json(result)
                
    elif args.image:
        # Test from image command line
        result = examiner.test_from_image(args.image, args.test)
        if result:
            examiner.print_result(result)
            
            if args.save:
                if args.format == 'txt' or args.format == 'both':
                    examiner.save_result_to_file(result)
                if args.format == 'json' or args.format == 'both':
                    examiner.save_result_as_json(result)
    else:
        # No arguments, show help
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK TEST EXAMPLES WITH SAVE:")
        print("=" * 60)
        print("python test_water_quality.py --rgb 255 0 0 --test ph --save")
        print("python test_water_quality.py --rgb 255 255 200 --test iron --save --format json")
        print("python test_water_quality.py --rgb 255 192 203 --test nitrate --save --format both")
        print("python test_water_quality.py --image test_strip.jpg --test ph --save")
        print("python test_water_quality.py -i  # Interactive mode")