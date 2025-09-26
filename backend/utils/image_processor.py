import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import os
from .serialization import to_py

class ImageProcessor:
    """
    Image processing class for extracting body structure features from animal images.
    Uses OpenCV for computer vision tasks.
    """
    
    def __init__(self):
        self.ready = True
        
        # Initialize feature extractors
        self.contour_detector = cv2.createLineSegmentDetector()
        
        # Define feature extraction parameters
        self.features_config = {
            'cow': {
                'body_length_ratio': 1.4,  # Typical body length to height ratio
                'chest_width_ratio': 0.7,   # Chest width to body height ratio
                'leg_length_ratio': 0.5,    # Leg length to body height ratio
            },
            'buffalo': {
                'body_length_ratio': 1.3,
                'chest_width_ratio': 0.75,
                'leg_length_ratio': 0.48,
            }
        }
    
    def is_ready(self) -> bool:
        """Check if the image processor is ready."""
        return self.ready
    
    def preprocess_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Preprocess the input image for feature extraction.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Resize image to standard size while maintaining aspect ratio
            height, width = image.shape[:2]
            target_width = 800
            target_height = int(height * (target_width / width))
            
            image = cv2.resize(image, (target_width, target_height))
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            return {
                'original': image,
                'gray': gray,
                'hsv': hsv,
                'width': target_width,
                'height': target_height
            }
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_animal_contour(self, processed_image: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract the main animal contour from the image.
        
        Args:
            processed_image: Dictionary containing preprocessed images
            
        Returns:
            Main contour of the animal
        """
        try:
            gray = processed_image['gray']
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold to handle varying lighting
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest contour (assuming it's the animal)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter out small contours
            if cv2.contourArea(largest_contour) < 1000:
                return None
            
            return largest_contour
            
        except Exception as e:
            print(f"Error extracting contour: {e}")
            return None
    
    def calculate_body_measurements(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculate body measurements from the animal contour.
        
        Args:
            contour: Animal contour
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary of body measurements
        """
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate basic measurements
            body_length = w
            body_height = h
            body_area = cv2.contourArea(contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Find extreme points
            leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
            
            # Calculate chest width (approximate as 1/3 from the front)
            chest_x = leftmost[0] + int(w * 0.33)
            chest_points = [point[0] for point in contour[:, 0] if abs(point[0] - chest_x) < 10]
            
            if len(chest_points) >= 2:
                chest_width = max(chest_points) - min(chest_points)
            else:
                chest_width = h * 0.7  # Fallback estimate
            
            # Calculate leg measurements (bottom 40% of the animal)
            leg_region_start = y + int(h * 0.6)
            leg_points = [point[0] for point in contour[:, 0] if point[1] >= leg_region_start]
            
            if leg_points:
                leg_span = max(leg_points) - min(leg_points)
                leg_length = h - leg_region_start
            else:
                leg_span = w * 0.8
                leg_length = h * 0.4
            
            # Calculate relative measurements (normalized by image size)
            image_height, image_width = image_shape[:2]
            
            measurements = {
                'body_length': body_length,
                'body_height': body_height,
                'body_area': body_area,
                'aspect_ratio': aspect_ratio,
                'chest_width': chest_width,
                'leg_span': leg_span,
                'leg_length': leg_length,
                'body_length_normalized': body_length / image_width,
                'body_height_normalized': body_height / image_height,
                'chest_width_normalized': chest_width / body_height if body_height > 0 else 0,
                'leg_length_normalized': leg_length / body_height if body_height > 0 else 0,
                'coordinates': {
                    'bounding_box': [x, y, w, h],
                    'leftmost': leftmost,
                    'rightmost': rightmost,
                    'topmost': topmost,
                    'bottommost': bottommost
                }
            }
            
            return measurements
            
        except Exception as e:
            print(f"Error calculating measurements: {e}")
            return {}
    
    def calculate_structural_features(self, measurements: Dict[str, Any], species: str) -> Dict[str, Any]:
        """
        Calculate structural features relevant for productivity classification.
        
        Args:
            measurements: Basic body measurements
            species: Animal species ('cow' or 'buffalo')
            
        Returns:
            Dictionary of structural features
        """
        try:
            if not measurements:
                return {}
            
            config = self.features_config.get(species, self.features_config['cow'])
            
            # Body proportions
            body_length_ratio = measurements.get('aspect_ratio', 0)
            chest_depth_ratio = measurements.get('chest_width_normalized', 0)
            leg_length_ratio = measurements.get('leg_length_normalized', 0)
            
            # Productivity indicators
            body_condition_score = self._calculate_body_condition(measurements)
            structural_soundness = self._calculate_structural_soundness(measurements)
            udder_placement_score = self._estimate_udder_placement(measurements)
            
            # Overall scores
            productivity_indicators = {
                'body_proportions': self._score_body_proportions(measurements, config),
                'chest_capacity': self._score_chest_capacity(measurements),
                'leg_structure': self._score_leg_structure(measurements),
                'overall_conformation': self._score_overall_conformation(measurements)
            }
            
            features = {
                'body_length_ratio': body_length_ratio,
                'chest_depth_ratio': chest_depth_ratio,
                'leg_length_ratio': leg_length_ratio,
                'body_condition_score': body_condition_score,
                'structural_soundness': structural_soundness,
                'udder_placement_score': udder_placement_score,
                'productivity_indicators': productivity_indicators,
                'measurements': measurements
            }
            
            return features
            
        except Exception as e:
            print(f"Error calculating structural features: {e}")
            return {}
    
    def _calculate_body_condition(self, measurements: Dict[str, Any]) -> float:
        """Calculate body condition score based on measurements."""
        try:
            aspect_ratio = measurements.get('aspect_ratio', 0)
            chest_ratio = measurements.get('chest_width_normalized', 0)
            
            # Ideal ratios for good body condition
            ideal_aspect = 1.4
            ideal_chest = 0.7
            
            aspect_score = 1.0 - abs(aspect_ratio - ideal_aspect) / ideal_aspect
            chest_score = 1.0 - abs(chest_ratio - ideal_chest) / ideal_chest
            
            return max(0, min(1, (aspect_score + chest_score) / 2))
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _calculate_structural_soundness(self, measurements: Dict[str, Any]) -> float:
        """Calculate structural soundness score."""
        try:
            leg_ratio = measurements.get('leg_length_normalized', 0)
            
            # Ideal leg length ratio
            ideal_leg_ratio = 0.5
            
            leg_score = 1.0 - abs(leg_ratio - ideal_leg_ratio) / ideal_leg_ratio
            
            return max(0, min(1, leg_score))
            
        except Exception:
            return 0.5
    
    def _estimate_udder_placement(self, measurements: Dict[str, Any]) -> float:
        """Estimate udder placement score based on body structure."""
        try:
            # This is a simplified estimation based on overall body proportions
            body_condition = self._calculate_body_condition(measurements)
            structural_soundness = self._calculate_structural_soundness(measurements)
            
            return (body_condition + structural_soundness) / 2
            
        except Exception:
            return 0.5
    
    def _score_body_proportions(self, measurements: Dict[str, Any], config: Dict[str, Any]) -> float:
        """Score body proportions for productivity."""
        try:
            aspect_ratio = measurements.get('aspect_ratio', 0)
            ideal_ratio = config['body_length_ratio']
            
            score = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
            return max(0, min(1, score))
            
        except Exception:
            return 0.5
    
    def _score_chest_capacity(self, measurements: Dict[str, Any]) -> float:
        """Score chest capacity for productivity."""
        try:
            chest_ratio = measurements.get('chest_width_normalized', 0)
            
            # Higher chest capacity is generally better
            score = min(1.0, chest_ratio / 0.8)
            return max(0, score)
            
        except Exception:
            return 0.5
    
    def _score_leg_structure(self, measurements: Dict[str, Any]) -> float:
        """Score leg structure for productivity."""
        try:
            leg_ratio = measurements.get('leg_length_normalized', 0)
            
            # Moderate leg length is optimal
            ideal_leg_ratio = 0.5
            score = 1.0 - abs(leg_ratio - ideal_leg_ratio) / ideal_leg_ratio
            
            return max(0, min(1, score))
            
        except Exception:
            return 0.5
    
    def _score_overall_conformation(self, measurements: Dict[str, Any]) -> float:
        """Score overall conformation."""
        try:
            body_condition = self._calculate_body_condition(measurements)
            structural_soundness = self._calculate_structural_soundness(measurements)
            
            return (body_condition + structural_soundness) / 2
            
        except Exception:
            return 0.5
    
    def extract_features(self, image_path: str, species: str) -> Optional[Dict[str, Any]]:
        """
        Main method to extract all features from an animal image.
        
        Args:
            image_path: Path to the animal image
            species: Animal species ('cow' or 'buffalo')
            
        Returns:
            Dictionary containing all extracted features
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return None
            
            # Extract animal contour
            contour = self.extract_animal_contour(processed_image)
            if contour is None:
                return None
            
            # Calculate measurements
            measurements = self.calculate_body_measurements(
                contour, 
                (processed_image['height'], processed_image['width'])
            )
            
            if not measurements:
                return None
            
            # Calculate structural features
            features = self.calculate_structural_features(measurements, species)
            
            if features:
                # Add metadata
                features['species'] = species
                features['image_dimensions'] = {
                    'width': processed_image['width'],
                    'height': processed_image['height']
                }
                features['processing_success'] = True

                # Ensure JSON-safe output
                return to_py(features)
            else:
                return None
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None