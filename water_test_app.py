import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from image_processor import ImageProcessor
from ml_classifier import ColorClassifier

class WaterTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Water Quality Test Analyzer")
        self.root.geometry("1400x800")
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.ml_classifier = ColorClassifier()
        
        # Try to load pre-trained model, or train if not available
        self.initialize_classifier()
        
        # Variables
        self.before_image = None
        self.after_image = None
        self.before_image_path = None
        self.after_image_path = None
        self.before_processed = None
        self.after_processed = None
        self.test_type = tk.StringVar(value="ph")
        self.result_data = None
        
        # Color reference database with chlorine
        self.color_references = self.load_color_references()
        
        # Setup UI
        self.setup_ui()
        
    def initialize_classifier(self):
        """Initialize the ML classifier"""
        try:
            if os.path.exists("color_classifier.pkl"):
                self.ml_classifier.load_model("color_classifier.pkl")
                print("Loaded pre-trained classifier")
            else:
                print("Training new classifier...")
                self.ml_classifier.train()
                self.ml_classifier.save_model("color_classifier.pkl")
                print("Classifier trained and saved")
        except Exception as e:
            print(f"Classifier initialization warning: {e}")
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
    
    def setup_ui(self):
        """Setup the enhanced user interface"""
        # Create main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_container, text="Water Quality Test Analyzer", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Test Type Selection with chlorine
        test_frame = ttk.LabelFrame(main_container, text="Select Test Type", padding="10")
        test_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        
        tests = [
            ("pH Test", "ph"),
            ("Iron Test", "iron"), 
            ("Nitrate Test", "nitrate"),
            ("Chlorine Test", "chlorine")
        ]
        
        for i, (label, value) in enumerate(tests):
            ttk.Radiobutton(test_frame, text=label, variable=self.test_type, 
                           value=value).grid(row=0, column=i, padx=10)
        
        # Image Capture Section
        image_frame = ttk.Frame(main_container)
        image_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        # Before Test Image
        before_frame = ttk.LabelFrame(image_frame, text="Before Test (Dry Strip)", padding="10")
        before_frame.grid(row=0, column=0, padx=10)
        
        self.before_label = ttk.Label(before_frame, text="No image loaded", 
                                     width=35, height=12, relief="solid")
        self.before_label.grid(row=0, column=0, pady=5)
        
        btn_frame1 = ttk.Frame(before_frame)
        btn_frame1.grid(row=1, column=0, pady=5)
        ttk.Button(btn_frame1, text="Load Image", 
                  command=lambda: self.load_image('before')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame1, text="Capture", 
                  command=lambda: self.capture_image('before')).pack(side=tk.LEFT, padx=5)
        
        # After Test Image
        after_frame = ttk.LabelFrame(image_frame, text="After Test (Wet Strip)", padding="10")
        after_frame.grid(row=0, column=1, padx=10)
        
        self.after_label = ttk.Label(after_frame, text="No image loaded", 
                                    width=35, height=12, relief="solid")
        self.after_label.grid(row=0, column=0, pady=5)
        
        btn_frame2 = ttk.Frame(after_frame)
        btn_frame2.grid(row=1, column=0, pady=5)
        ttk.Button(btn_frame2, text="Load Image", 
                  command=lambda: self.load_image('after')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame2, text="Capture", 
                  command=lambda: self.capture_image('after')).pack(side=tk.LEFT, padx=5)
        
        # Color Reference Display
        ref_frame = ttk.LabelFrame(image_frame, text="Color Reference", padding="10")
        ref_frame.grid(row=0, column=2, padx=10)
        
        self.ref_canvas = tk.Canvas(ref_frame, width=200, height=300, bg='white')
        self.ref_canvas.grid(row=0, column=0)
        self.update_color_reference()
        
        # Analyze Button
        analyze_btn = ttk.Button(main_container, text="ANALYZE WATER SAMPLE", 
                                command=self.analyze_sample,
                                style="Accent.TButton")
        analyze_btn.grid(row=3, column=0, columnspan=4, pady=20)
        
        # Results Section
        self.results_frame = ttk.LabelFrame(main_container, text="Analysis Results", padding="10")
        self.results_frame.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        results_tab = ttk.Frame(self.notebook)
        self.notebook.add(results_tab, text="Results")
        
        self.results_text = tk.Text(results_tab, height=15, width=90, font=('Courier', 10))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(results_tab, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text['yscrollcommand'] = scrollbar.set
        
        # Statistics tab
        stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(stats_tab, text="Statistics")
        
        self.stats_text = tk.Text(stats_tab, height=15, width=90, font=('Courier', 10))
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        stats_scrollbar = ttk.Scrollbar(stats_tab, command=self.stats_text.yview)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text['yscrollcommand'] = stats_scrollbar.set
        
        # Export buttons
        button_frame = ttk.Frame(main_container)
        button_frame.grid(row=5, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="Export JSON", 
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Report", 
                  command=self.save_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="New Test", 
                  command=self.reset_test).pack(side=tk.LEFT, padx=5)
    
    def update_color_reference(self):
        """Update color reference display based on test type"""
        self.ref_canvas.delete("all")
        test_type = self.test_type.get()
        colors = list(self.color_references[test_type].values())
        
        y = 10
        for color in colors[:8]:  # Show first 8 colors
            # Get approximate RGB for display (simplified)
            if 'yellow' in color['color'].lower():
                rgb = [255, 255, 0]
            elif 'red' in color['color'].lower():
                rgb = [255, 0, 0]
            elif 'orange' in color['color'].lower():
                rgb = [255, 165, 0]
            elif 'pink' in color['color'].lower():
                rgb = [255, 192, 203]
            elif 'purple' in color['color'].lower():
                rgb = [128, 0, 128]
            elif 'brown' in color['color'].lower():
                rgb = [139, 69, 19]
            else:
                rgb = [128, 128, 128]
            
            color_hex = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
            self.ref_canvas.create_rectangle(10, y, 190, y+25, fill=color_hex, outline='black')
            self.ref_canvas.create_text(100, y+12, 
                                      text=f"{color['color']} ({color['value']}{color['unit']})", 
                                      fill='white' if sum(rgb) < 384 else 'black')
            y += 30
    
    def load_image(self, image_type):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.display_image(file_path, image_type)
            if image_type == 'before':
                self.before_image_path = file_path
                self.before_image = cv2.imread(file_path)
                self.before_processed = self.image_processor.preprocess_image(file_path)
            else:
                self.after_image_path = file_path
                self.after_image = cv2.imread(file_path)
                self.after_processed = self.image_processor.preprocess_image(file_path)
    
    def capture_image(self, image_type):
        """Capture image from camera"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        cv2.namedWindow("Camera - Press SPACE to capture, ESC to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Draw guide box
            cv2.rectangle(frame, 
                         (center_x - 150, center_y - 150),
                         (center_x + 150, center_y + 150),
                         (0, 255, 0), 2)
            
            cv2.putText(frame, "Place test strip here", 
                       (center_x - 120, center_y - 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera - Press SPACE to capture, ESC to cancel", frame)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 32:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_{image_type}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                
                self.display_image(filename, image_type)
                if image_type == 'before':
                    self.before_image_path = filename
                    self.before_image = frame
                    self.before_processed = self.image_processor.preprocess_image(filename)
                else:
                    self.after_image_path = filename
                    self.after_image = frame
                    self.after_processed = self.image_processor.preprocess_image(filename)
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def display_image(self, image_path, image_type):
        """Display image in the GUI"""
        image = Image.open(image_path)
        image.thumbnail((350, 250))
        photo = ImageTk.PhotoImage(image)
        
        if image_type == 'before':
            self.before_label.config(image=photo, text="")
            self.before_label.image = photo
        else:
            self.after_label.config(image=photo, text="")
            self.after_label.image = photo
    
    def analyze_sample(self):
        """Main analysis function using ML and image processing"""
        if self.before_image is None or self.after_image is None:
            messagebox.showerror("Error", "Please load both before and after images")
            return
        
        try:
            # Show progress
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, "Analyzing samples...\n")
            self.root.update()
            
            # Extract test regions
            before_region = self.extract_test_region(self.before_processed)
            after_region = self.extract_test_region(self.after_processed)
            
            # Analyze colors using image processor
            before_colors = self.image_processor.analyze_color_distribution(before_region)
            after_colors = self.image_processor.analyze_color_distribution(after_region)
            
            # Get dominant colors
            before_dominant = before_colors[0]['rgb'] if before_colors else [128, 128, 128]
            after_dominant = after_colors[0]['rgb'] if after_colors else [128, 128, 128]
            
            # Get color names
            before_color_name = self.rgb_to_color_name(before_dominant)
            after_color_name = self.rgb_to_color_name(after_dominant)
            
            # Use ML classifier for color prediction
            ml_result = None
            try:
                if self.ml_classifier.is_trained:
                    ml_result = self.ml_classifier.predict_color(after_dominant)
            except:
                pass
            
            # Get color statistics
            before_stats = self.image_processor.calculate_color_statistics(before_region)
            after_stats = self.image_processor.calculate_color_statistics(after_region)
            
            # Compare images
            comparison = self.image_processor.compare_images(
                before_region, 
                after_region
            )
            
            # Detect color change regions
            change_analysis = self.image_processor.detect_color_change_region(
                before_region, 
                after_region
            )
            
            # SPECIAL CASE: For Iron test, if colors are the same (yellow shades), set to "None"
            current_test = self.test_type.get()
            
            if current_test == 'iron':
                # Check if both colors are yellow shades
                yellow_shades = ['yellow', 'light_yellow', 'pale_yellow', 'golden_yellow', 'deep_yellow', 'very_light_yellow']
                if before_color_name in yellow_shades and after_color_name in yellow_shades:
                    # Calculate color difference to be sure
                    color_diff = np.linalg.norm(np.array(before_dominant) - np.array(after_dominant))
                    
                    # If difference is small, consider it as no change
                    if color_diff < 60:  # Threshold for considering as same color
                        result = {
                            'color': 'Yellow (No Change)',
                            'level': 'None',
                            'value': 0.0,
                            'unit': 'mg/L',
                            'explanation': "The test strip shows no significant color change from the original dry strip, indicating no detectable iron in the water sample. Iron ions would normally form a colored complex with the reagent, but the absence of color change suggests iron concentration is below detectable levels (<0.05 mg/L).",
                            'recommendation': "✅ No detectable iron. Water is safe for consumption. Continue regular testing to monitor water quality."
                        }
                        
                        # Store comprehensive results
                        self.result_data = {
                            'test_type': current_test,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'before': {
                                'dominant_color': before_dominant,
                                'color_name': before_color_name,
                                'statistics': before_stats,
                                'color_distribution': before_colors
                            },
                            'after': {
                                'dominant_color': after_dominant,
                                'color_name': after_color_name,
                                'statistics': after_stats,
                                'color_distribution': after_colors
                            },
                            'comparison': {
                                'color_difference': float(comparison['total_difference']),
                                'mse': float(comparison['mse']),
                                'histogram_similarity': float(comparison['histogram_difference']),
                                'change_percentage': float(change_analysis['change_percentage']),
                                'mean_change': float(change_analysis['mean_change'])
                            },
                            'result': result,
                            'ml_analysis': ml_result,
                            'no_change_detected': True
                        }
                        
                        # Display results
                        self.display_results()
                        return
            
            # Normal flow for other cases
            # Match color to reference
            result = self.match_color_to_reference(after_color_name, after_dominant)
            
            # Add ML confidence if available
            if ml_result:
                result['ml_confidence'] = ml_result['confidence']
                result['ml_color'] = ml_result['color_name']
            
            # Add scientific explanation
            result['explanation'] = self.get_scientific_explanation(
                current_test, 
                result['color'],
                result['value']
            )
            
            # Add safety recommendation
            result['recommendation'] = self.get_safety_recommendation(
                current_test,
                result['level'],
                result['value']
            )
            
            # Store comprehensive results
            self.result_data = {
                'test_type': current_test,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'before': {
                    'dominant_color': before_dominant,
                    'color_name': before_color_name,
                    'statistics': before_stats,
                    'color_distribution': before_colors
                },
                'after': {
                    'dominant_color': after_dominant,
                    'color_name': after_color_name,
                    'statistics': after_stats,
                    'color_distribution': after_colors
                },
                'comparison': {
                    'color_difference': float(comparison['total_difference']),
                    'mse': float(comparison['mse']),
                    'histogram_similarity': float(comparison['histogram_difference']),
                    'change_percentage': float(change_analysis['change_percentage']),
                    'mean_change': float(change_analysis['mean_change'])
                },
                'result': result,
                'ml_analysis': ml_result,
                'no_change_detected': False
            }
            
            # Display results
            self.display_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def extract_test_region(self, processed_image):
        """Extract test strip region using image processor"""
        if processed_image is None:
            return None
        
        # Detect test strip
        strip_region = self.image_processor.detect_test_strip(processed_image)
        
        # Extract region
        region = self.image_processor.extract_test_region(processed_image, strip_region)
        
        return region
    
    def rgb_to_color_name(self, rgb):
        """Convert RGB to color name with enhanced detection"""
        r, g, b = rgb
        
        # Enhanced color classification for chlorine (yellows)
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
    
    def match_color_to_reference(self, color_name, rgb_values=None):
        """Match detected color to reference values"""
        test_type = self.test_type.get()
        reference = self.color_references[test_type]
        
        # Direct match
        if color_name in reference:
            ref_data = reference[color_name]
            return {
                'color': ref_data['color'],
                'level': ref_data['level'],
                'value': ref_data['value'],
                'unit': ref_data['unit']
            }
        
        # Try to find closest match if RGB values provided
        if rgb_values is not None:
            closest_match = self.find_closest_color(rgb_values, reference)
            if closest_match:
                return closest_match
        
        # Default fallback
        default_key = list(reference.keys())[len(reference)//2]  # Middle value
        ref_data = reference[default_key]
        return {
            'color': ref_data['color'],
            'level': ref_data['level'],
            'value': ref_data['value'],
            'unit': ref_data['unit']
        }
    
    def find_closest_color(self, rgb_values, reference):
        """Find closest color in reference based on RGB distance"""
        min_distance = float('inf')
        best_match = None
        
        # Simplified RGB approximations for reference colors
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
            'ph': f"The color changed to {color} because pH indicator molecules gain or lose protons (H⁺) depending on the solution's acidity. This changes the molecular structure, altering how light is absorbed, resulting in the observed color. The measured pH value of {value} indicates the concentration of hydrogen ions in the water.",
            
            'iron': f"The appearance of a {color} color indicates iron presence. Iron ions form a coordination complex with the reagent on the test strip. This new complex has a different molecular structure that absorbs specific wavelengths of light, appearing as a {color.lower()} color. The concentration of {value} mg/L represents the amount of iron in the sample.",
            
            'nitrate': f"The development of a {color} color shows nitrate presence. Nitrate is chemically reduced to nitrite, which then participates in the Griess reaction to form a highly colored azo compound. The intensity of the {color.lower()} color is directly proportional to the nitrate concentration of {value} mg/L.",
            
            'chlorine': f"The appearance of a {color} color indicates the presence of free chlorine. The DPD (N,N-diethyl-p-phenylenediamine) indicator reacts with chlorine to form a pinkish-yellow compound called Würster dye. The intensity of the yellow color is directly proportional to the chlorine concentration of {value} mg/L. Free chlorine reacts immediately, while combined chlorine reacts more slowly."
        }
        
        return explanations.get(test_type, "Color change indicates presence of the target analyte.")
    
    def get_safety_recommendation(self, test_type, level, value):
        """Get safety recommendations based on results"""
        recommendations = {
            'ph': {
                'Strong Acid': "⚠️ CRITICAL: Strongly acidic water (pH < 3) can cause severe corrosion of pipes and may leach toxic metals. Do not consume. Immediate water treatment required with pH adjustment system.",
                'Acidic': "⚠️ Acidic water (pH 3-5) may corrode pipes and leach metals like lead and copper. Consider installing an acid-neutralizing filter.",
                'Slightly Acidic': "⚠️ Slightly acidic water (pH 5-6.5) may cause minor corrosion. Monitor regularly and consider pH adjustment if problems occur.",
                'Neutral': "✅ pH level is within safe drinking water range (6.5-8.5). Water is suitable for consumption.",
                'Slightly Basic': "⚠️ Slightly basic water (pH 7.5-8.5) may cause scale formation but is generally safe for consumption.",
                'Basic': "⚠️ Basic water (pH > 8.5) may cause scale formation and taste issues. Consider water softening if problems persist."
            },
            'iron': {
                'None': "✅ Iron level is within acceptable limits (<0.3 mg/L). No action required.",
                'Low': "✅ Iron level is low (0.1-0.3 mg/L). Water is safe but may cause minor staining over time.",
                'Medium': "⚠️ Moderate iron detected (0.3-1.0 mg/L). May cause metallic taste and staining. Consider iron filtration system.",
                'High': "⚠️ High iron level (1.0-3.0 mg/L). Will cause staining, metallic taste, and may promote bacterial growth. Iron filtration recommended.",
                'Very High': "⚠️ Very high iron (3.0-5.0 mg/L). Significant staining and taste issues. Immediate water treatment required.",
                'Severe': "⚠️ CRITICAL: Severe iron contamination (>5.0 mg/L). Water is unsafe for consumption. Professional water treatment system mandatory."
            },
            'nitrate': {
                'None': "✅ Nitrate level is within safe limits (<1 mg/L). Water is safe for all uses.",
                'Low': "✅ Nitrate level is low (1-5 mg/L). Water is safe for consumption.",
                'Medium': "⚠️ Moderate nitrate detected (5-10 mg/L). Safe for adults but caution advised for infants and pregnant women.",
                'High': "⚠️ High nitrate level (10-20 mg/L). Unsafe for infants and pregnant women. Adults should limit consumption. Consider alternative water source.",
                'Very High': "⚠️ CRITICAL: Very high nitrate (20-50 mg/L). Water is unsafe for consumption. Do not use for drinking or cooking.",
                'Severe': "⚠️ DANGEROUS: Severe nitrate contamination (>50 mg/L). Water is toxic. Seek alternative water source immediately."
            },
            'chlorine': {
                'Very Low': "⚠️ Chlorine level is very low (0-0.5 mg/L). May not provide adequate disinfection. Consider increasing chlorine dosage for proper water treatment.",
                'Low': "⚠️ Chlorine level is low (0.5-1.0 mg/L). Minimal disinfection capability. Monitor closely and adjust as needed.",
                'Low-Moderate': "✅ Chlorine level is acceptable (1.0-2.0 mg/L). Provides adequate disinfection for most applications.",
                'Moderate': "✅ Optimal chlorine level (2.0-3.0 mg/L). Provides effective disinfection while minimizing taste and odor issues.",
                'Moderate-High': "⚠️ Chlorine level is elevated (3.0-5.0 mg/L). May cause noticeable taste and odor but still safe for consumption.",
                'High': "⚠️ High chlorine level (5.0-7.0 mg/L). May cause strong taste and odor. Can cause skin and eye irritation. Consider reducing chlorine dosage.",
                'Very High': "⚠️ CRITICAL: Very high chlorine (>7.0 mg/L). Water may be unsafe for consumption. Can cause respiratory irritation and damage to plumbing. Immediately reduce chlorine levels."
            }
        }
        
        return recommendations.get(test_type, {}).get(level, "No specific recommendation available.")
    
    def display_results(self):
        """Display analysis results in the text widgets"""
        if not self.result_data:
            return
        
        # Clear both text widgets
        self.results_text.delete(1.0, tk.END)
        self.stats_text.delete(1.0, tk.END)
        
        # Format main results
        result = self.result_data['result']
        
        output = []
        output.append("=" * 80)
        output.append(f"WATER QUALITY TEST RESULTS - {self.test_type.get().upper()}")
        output.append("=" * 80)
        output.append(f"\nDate/Time: {self.result_data['timestamp']}")
        output.append("\n" + "-" * 80)
        
        # Test results - Modified for no change detection
        output.append("\nTEST RESULTS:")
        
        if self.result_data.get('no_change_detected', False):
            output.append(f"  • Status: NO SIGNIFICANT COLOR CHANGE DETECTED")
            output.append(f"  • Before Color: {self.result_data['before']['color_name'].title()}")
            output.append(f"  • After Color: {self.result_data['after']['color_name'].title()}")
            output.append(f"  • Result: No detectable iron in sample")
            output.append(f"  • Concentration: < 0.05 mg/L (Below detectable limit)")
        else:
            output.append(f"  • Detected Color: {result['color']}")
            output.append(f"  • Concentration: {result['value']} {result['unit']}")
            output.append(f"  • Level: {result['level']}")
        
        # ML confidence if available
        if 'ml_confidence' in result:
            output.append(f"  • ML Confidence: {result['ml_confidence']*100:.1f}%")
        
        # Add drinking water standards for chlorine
        if self.test_type.get() == 'chlorine':
            output.append("\nDRINKING WATER STANDARDS:")
            output.append("  • WHO Guideline: ≤ 5.0 mg/L")
            output.append("  • EPA Maximum: 4.0 mg/L")
            output.append("  • Ideal Range: 0.5 - 2.0 mg/L")
        
        output.append("\n" + "-" * 80)
        
        # Scientific explanation
        output.append("\nSCIENTIFIC EXPLANATION:")
        output.append(f"  {result['explanation']}")
        
        output.append("\n" + "-" * 80)
        
        # Safety recommendation
        output.append("\nSAFETY RECOMMENDATION:")
        output.append(f"  {result['recommendation']}")
        
        output.append("\n" + "-" * 80)
        
        # Basic color analysis
        output.append("\nCOLOR ANALYSIS:")
        output.append(f"  • Before Test RGB: {self.result_data['before']['dominant_color']}")
        output.append(f"  • After Test RGB: {self.result_data['after']['dominant_color']}")
        output.append(f"  • Color Difference: {self.result_data['comparison']['color_difference']:.2f}")
        
        output.append("\n" + "=" * 80)
        
        self.results_text.insert(1.0, "\n".join(output))
        
        # Format statistics
        stats_output = []
        stats_output.append("=" * 80)
        stats_output.append("DETAILED STATISTICAL ANALYSIS")
        stats_output.append("=" * 80)
        
        # Color distribution
        stats_output.append("\nCOLOR DISTRIBUTION - BEFORE TEST:")
        for i, color in enumerate(self.result_data['before']['color_distribution']):
            stats_output.append(f"  Color {i+1}: RGB{color['rgb']} - {color['percentage']:.1f}%")
        
        stats_output.append("\nCOLOR DISTRIBUTION - AFTER TEST:")
        for i, color in enumerate(self.result_data['after']['color_distribution']):
            stats_output.append(f"  Color {i+1}: RGB{color['rgb']} - {color['percentage']:.1f}%")
        
        # Image comparison metrics
        stats_output.append("\nIMAGE COMPARISON METRICS:")
        stats_output.append(f"  • Mean Squared Error: {self.result_data['comparison']['mse']:.2f}")
        stats_output.append(f"  • Histogram Similarity: {self.result_data['comparison']['histogram_similarity']:.2f}")
        stats_output.append(f"  • Change Percentage: {self.result_data['comparison']['change_percentage']:.1f}%")
        stats_output.append(f"  • Mean Change Intensity: {self.result_data['comparison']['mean_change']:.2f}")
        
        # Color statistics
        stats_output.append("\nBEFORE TEST STATISTICS:")
        before_stats = self.result_data['before']['statistics']
        stats_output.append(f"  • Mean RGB: {before_stats['rgb']['mean']}")
        stats_output.append(f"  • Std Dev RGB: {before_stats['rgb']['std']}")
        stats_output.append(f"  • HSV Mean: {before_stats['hsv']['mean']}")
        
        stats_output.append("\nAFTER TEST STATISTICS:")
        after_stats = self.result_data['after']['statistics']
        stats_output.append(f"  • Mean RGB: {after_stats['rgb']['mean']}")
        stats_output.append(f"  • Std Dev RGB: {after_stats['rgb']['std']}")
        stats_output.append(f"  • HSV Mean: {after_stats['hsv']['mean']}")
        
        # ML analysis if available
        if self.result_data['ml_analysis']:
            stats_output.append("\nMACHINE LEARNING ANALYSIS:")
            ml = self.result_data['ml_analysis']
            stats_output.append(f"  • Predicted Color: {ml['color_name']}")
            stats_output.append(f"  • Confidence: {ml['confidence']*100:.1f}%")
            stats_output.append("  • All Probabilities:")
            for color, prob in ml['all_probabilities'].items():
                stats_output.append(f"      {color}: {prob*100:.1f}%")
        
        stats_output.append("\n" + "=" * 80)
        
        self.stats_text.insert(1.0, "\n".join(stats_output))
    
    def export_results(self):
        """Export results to JSON file"""
        if not self.result_data:
            messagebox.showerror("Error", "No results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            # Convert numpy arrays to lists for JSON serialization
            export_data = self.convert_to_serializable(self.result_data)
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=4)
            messagebox.showinfo("Success", f"Results exported to {filename}")
    
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
    
    def save_report(self):
        """Save a formatted report to text file"""
        if not self.result_data:
            messagebox.showerror("Error", "No results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(self.results_text.get(1.0, tk.END))
                f.write("\n\n")
                f.write(self.stats_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Report saved to {filename}")
    
    def reset_test(self):
        """Reset the test for a new sample"""
        self.before_image = None
        self.after_image = None
        self.before_image_path = None
        self.after_image_path = None
        self.before_processed = None
        self.after_processed = None
        self.result_data = None
        
        # Reset image labels
        self.before_label.config(image="", text="No image loaded")
        self.after_label.config(image="", text="No image loaded")
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.stats_text.delete(1.0, tk.END)
        
        # Update color reference
        self.update_color_reference()
        
        messagebox.showinfo("Reset", "Ready for new test. Please load new images.")

def main():
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Accent.TButton", font=('Arial', 11, 'bold'))
    
    # Bind test type change event
    def on_test_type_change(*args):
        if hasattr(root, 'app'):
            root.app.update_color_reference()
    
    app = WaterTestApp(root)
    root.app = app  # Store reference for callbacks
    app.test_type.trace('w', on_test_type_change)
    
    root.mainloop()

if __name__ == "__main__":
    main()