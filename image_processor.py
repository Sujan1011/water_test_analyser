import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats

class ImageProcessor:
    def __init__(self):
        self.color_spaces = {
            'RGB': None,
            'HSV': None,
            'LAB': None
        }
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize if too large
        h, w = img.shape[:2]
        if max(h, w) > 1000:
            scale = 1000 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # Convert to different color spaces
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        return {
            'original': img,
            'rgb': rgb,
            'hsv': hsv,
            'lab': lab,
            'shape': img.shape
        }
    
    def detect_test_strip(self, image):
        """Detect the test strip region in the image"""
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image['original'], cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Combine threshold and edges
        combined = cv2.bitwise_or(thresh, edges)
        
        # Morphological operations to close gaps
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area and aspect ratio
        min_area = image['shape'][0] * image['shape'][1] * 0.01  # At least 1% of image
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Test strips typically have aspect ratio between 0.5 and 3
                if 0.5 < aspect_ratio < 3:
                    valid_contours.append(contour)
        
        if not valid_contours:
            return None
        
        # Find the largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'contour': largest_contour,
            'area': cv2.contourArea(largest_contour)
        }
    
    def extract_test_region(self, image, strip_region=None):
        """Extract the test region from the strip"""
        if image is None:
            return None
        
        if strip_region is None:
            # If strip not detected, use center of image with margin
            h, w = image['rgb'].shape[:2]
            margin = 0.2  # 20% margin
            x = int(w * margin)
            y = int(h * margin)
            w = int(w * (1 - 2*margin))
            h = int(h * (1 - 2*margin))
        else:
            x = max(0, strip_region['x'])
            y = max(0, strip_region['y'])
            w = min(strip_region['width'], image['rgb'].shape[1] - x)
            h = min(strip_region['height'], image['rgb'].shape[0] - y)
        
        # Extract region
        region = image['rgb'][y:y+h, x:x+w]
        
        # Ensure region is not empty
        if region.size == 0:
            # Fallback to center region
            h, w = image['rgb'].shape[:2]
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 3
            region = image['rgb'][center_y-size:center_y+size, 
                                 center_x-size:center_x+size]
        
        return region
    
    def enhance_region(self, region):
        """Enhance the test region for better color analysis"""
        if region is None or region.size == 0:
            return region
        
        # Apply CLAHE for better contrast
        lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def analyze_color_distribution(self, region, n_colors=3):
        """Analyze color distribution using K-means clustering"""
        if region is None or region.size == 0:
            return []
        
        # Enhance region
        enhanced = self.enhance_region(region)
        
        # Reshape image to be a list of pixels
        pixels = enhanced.reshape(-1, 3)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the dominant colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Count pixels in each cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        percentages = counts / counts.sum() * 100
        
        # Sort by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        
        dominant_colors = []
        for idx in sorted_indices:
            dominant_colors.append({
                'rgb': colors[idx].tolist(),
                'percentage': percentages[idx],
                'hex': '#{:02x}{:02x}{:02x}'.format(colors[idx][0], 
                                                    colors[idx][1], 
                                                    colors[idx][2])
            })
        
        return dominant_colors
    
    def calculate_color_statistics(self, region):
        """Calculate statistical properties of colors in the region"""
        if region is None or region.size == 0:
            return None
        
        # Calculate RGB statistics
        rgb_stats = {
            'mean': np.mean(region, axis=(0, 1)).tolist(),
            'std': np.std(region, axis=(0, 1)).tolist(),
            'median': np.median(region, axis=(0, 1)).tolist(),
            'min': np.min(region, axis=(0, 1)).tolist(),
            'max': np.max(region, axis=(0, 1)).tolist()
        }
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        hsv_stats = {
            'mean': np.mean(hsv, axis=(0, 1)).tolist(),
            'std': np.std(hsv, axis=(0, 1)).tolist()
        }
        
        # Calculate percentiles
        percentiles = {}
        for p in [25, 50, 75]:
            percentiles[p] = np.percentile(region, p, axis=(0, 1)).tolist()
        
        return {
            'rgb': rgb_stats,
            'hsv': hsv_stats,
            'percentiles': percentiles
        }
    
    def compare_images(self, img1, img2):
        """Compare two images and calculate differences"""
        if img1 is None or img2 is None:
            return None
        
        # Ensure same size
        if img1.shape != img2.shape:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        # Calculate absolute difference
        diff = cv2.absdiff(img1, img2)
        
        # Calculate mean difference per channel
        mean_diff = np.mean(diff, axis=(0, 1))
        
        # Calculate MSE
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # Calculate PSNR
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        
        # Calculate color histogram difference
        hist_diff = self.compare_histograms(img1, img2)
        
        # Calculate structural similarity (simplified)
        ssim = 1 / (1 + np.sqrt(mse) / 255.0)
        
        return {
            'mean_difference': mean_diff.tolist(),
            'total_difference': float(np.mean(mean_diff)),
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'histogram_difference': float(hist_diff),
            'difference_image': diff
        }
    
    def compare_histograms(self, img1, img2):
        """Compare color histograms of two images"""
        hist_similarity = 0
        
        for i in range(3):  # For each channel
            hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            
            # Normalize histograms
            hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
            hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2
            
            # Calculate correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            hist_similarity += correlation
        
        return hist_similarity / 3  # Average across channels
    
    def detect_color_change_region(self, before, after):
        """Detect regions where color changed significantly"""
        if before is None or after is None:
            return None
        
        # Ensure same size
        if before.shape != after.shape:
            h = min(before.shape[0], after.shape[0])
            w = min(before.shape[1], after.shape[1])
            before = cv2.resize(before, (w, h))
            after = cv2.resize(after, (w, h))
        
        # Convert to float for calculation
        before_float = before.astype(float)
        after_float = after.astype(float)
        
        # Calculate color difference in LAB space for perceptual uniformity
        before_lab = cv2.cvtColor(before, cv2.COLOR_RGB2LAB).astype(float)
        after_lab = cv2.cvtColor(after, cv2.COLOR_RGB2LAB).astype(float)
        
        # Calculate Delta E (color difference)
        delta_e = np.sqrt(np.sum((before_lab - after_lab) ** 2, axis=2))
        
        # Find regions with significant change
        threshold = np.mean(delta_e) + np.std(delta_e)
        significant_change = delta_e > threshold
        
        # Create mask of changed regions
        change_mask = significant_change.astype(np.uint8) * 255
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate change statistics
        change_percentage = np.sum(change_mask > 0) / change_mask.size * 100
        mean_change = np.mean(delta_e[significant_change]) if np.any(significant_change) else 0
        max_change = np.max(delta_e) if delta_e.size > 0 else 0
        
        return {
            'mask': change_mask,
            'change_percentage': float(change_percentage),
            'mean_change': float(mean_change),
            'max_change': float(max_change),
            'delta_e_map': delta_e
        }
    
    def create_color_visualization(self, image, colors):
        """Create a visualization of color analysis"""
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Create a color bar showing dominant colors
        bar_height = 50
        color_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
        
        start_x = 0
        for color_info in colors:
            color_width = int(w * color_info['percentage'] / 100)
            if color_width > 0:
                color_bar[:, start_x:start_x+color_width] = color_info['rgb']
                start_x += color_width
        
        # Combine original image with color bar
        visualization = np.vstack([image, color_bar])
        
        return visualization
    
    def extract_color_features(self, region):
        """Extract comprehensive color features for ML"""
        if region is None or region.size == 0:
            return None
        
        features = []
        
        # Color statistics
        stats = self.calculate_color_statistics(region)
        if stats:
            features.extend(stats['rgb']['mean'])
            features.extend(stats['rgb']['std'])
            features.extend(stats['rgb']['median'])
            features.extend(stats['hsv']['mean'])
            features.extend(stats['hsv']['std'])
        
        # Color distribution
        distribution = self.analyze_color_distribution(region, n_colors=3)
        for color in distribution:
            features.extend(color['rgb'])
            features.append(color['percentage'])
        
        # Texture features (simplified)
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        features.append(np.mean(gray))
        features.append(np.std(gray))
        
        return np.array(features)

# Usage example
if __name__ == "__main__":
    processor = ImageProcessor()
    
    # Example usage with a test image
    image_path = "test_strip.jpg"  # Replace with actual path
    if os.path.exists(image_path):
        image_data = processor.preprocess_image(image_path)
        if image_data:
            strip_region = processor.detect_test_strip(image_data)
            test_region = processor.extract_test_region(image_data, strip_region)
            dominant_colors = processor.analyze_color_distribution(test_region)
            stats = processor.calculate_color_statistics(test_region)
            
            print("Dominant Colors:", dominant_colors)
            print("Color Statistics:", stats)
            
            # Create visualization
            vis = processor.create_color_visualization(test_region, dominant_colors)
            if vis is not None:
                cv2.imwrite("visualization.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))