import requests
import json
import time
import os
from datetime import datetime

class ATCTester:
    """Test class for the Animal Type Classification system."""
    
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.test_results = []
    
    def log_test(self, test_name, success, message="", data=None):
        """Log test results."""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}: {message}")
    
    def test_api_health(self):
        """Test API health endpoint."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("API Health Check", True, "API is healthy", data)
                    return True
                else:
                    self.log_test("API Health Check", False, f"API status: {data.get('status')}")
                    return False
            else:
                self.log_test("API Health Check", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API Health Check", False, f"Connection error: {str(e)}")
            return False
    
    def test_root_endpoint(self):
        """Test root API endpoint."""
        try:
            response = requests.get(f"{self.api_base_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "Animal Type Classification API" in data.get("message", ""):
                    self.log_test("Root Endpoint", True, "Root endpoint working")
                    return True
                else:
                    self.log_test("Root Endpoint", False, "Unexpected response format")
                    return False
            else:
                self.log_test("Root Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Root Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_statistics_endpoint(self):
        """Test statistics endpoint."""
        try:
            response = requests.get(f"{self.api_base_url}/stats", timeout=5)
            if response.status_code == 200:
                data = response.json()
                required_fields = ["total_classifications", "high_productivity_count", "average_score"]
                if all(field in data for field in required_fields):
                    self.log_test("Statistics Endpoint", True, "Statistics endpoint working", data)
                    return True
                else:
                    self.log_test("Statistics Endpoint", False, "Missing required fields in response")
                    return False
            else:
                self.log_test("Statistics Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Statistics Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_history_endpoint(self):
        """Test history endpoint."""
        try:
            response = requests.get(f"{self.api_base_url}/history?limit=5", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "history" in data and "count" in data:
                    self.log_test("History Endpoint", True, f"History endpoint working, {data['count']} records", data)
                    return True
                else:
                    self.log_test("History Endpoint", False, "Unexpected response format")
                    return False
            else:
                self.log_test("History Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("History Endpoint", False, f"Error: {str(e)}")
            return False
    
    def create_test_image(self, filename="test_animal.jpg"):
        """Create a simple test image for testing."""
        try:
            from PIL import Image, ImageDraw
            import numpy as np
            
            # Create a simple test image that resembles an animal silhouette
            width, height = 800, 600
            
            # Create a white background
            img = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw a simple animal-like shape (rectangle with rounded ends)
            body_left = width // 4
            body_right = 3 * width // 4
            body_top = height // 3
            body_bottom = 2 * height // 3
            
            # Main body
            draw.rectangle([body_left, body_top, body_right, body_bottom], fill='brown', outline='black')
            
            # Head (circle)
            head_size = 80
            head_x = body_left - head_size // 2
            head_y = body_top + (body_bottom - body_top) // 3
            draw.ellipse([head_x, head_y, head_x + head_size, head_y + head_size], fill='brown', outline='black')
            
            # Legs
            leg_width = 20
            leg_height = height // 6
            leg_positions = [
                body_left + 50,  # Front left
                body_left + 150, # Front right  
                body_right - 150, # Back left
                body_right - 50   # Back right
            ]
            
            for leg_x in leg_positions:
                draw.rectangle([leg_x, body_bottom, leg_x + leg_width, body_bottom + leg_height], 
                             fill='brown', outline='black')
            
            # Save the image
            img.save(filename)
            return filename
            
        except Exception as e:
            print(f"Could not create test image: {e}")
            return None
    
    def test_image_upload(self, image_path=None):
        """Test image upload and classification."""
        if image_path is None:
            # Create a test image
            image_path = self.create_test_image()
            if image_path is None:
                self.log_test("Image Upload", False, "Could not create test image")
                return False
        
        if not os.path.exists(image_path):
            self.log_test("Image Upload", False, f"Test image not found: {image_path}")
            return False
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path, f, 'image/jpeg')}
                data = {
                    'species': 'cow',
                    'animal_name': 'Test Cow'
                }
                
                response = requests.post(
                    f"{self.api_base_url}/upload",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success") and "classification" in result:
                    classification = result["classification"]
                    productivity_class = classification.get("productivity_class")
                    confidence = classification.get("confidence", 0)
                    
                    self.log_test("Image Upload", True, 
                                f"Classification successful: {productivity_class} ({confidence:.2%} confidence)",
                                result)
                    return True
                else:
                    self.log_test("Image Upload", False, "Classification failed or incomplete response")
                    return False
            else:
                error_msg = response.text if response.text else f"HTTP {response.status_code}"
                self.log_test("Image Upload", False, f"Upload failed: {error_msg}")
                return False
                
        except Exception as e:
            self.log_test("Image Upload", False, f"Error during upload: {str(e)}")
            return False
        finally:
            # Clean up test image
            if image_path == "test_animal.jpg" and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
    
    def run_all_tests(self):
        """Run all tests and generate a report."""
        print("=" * 60)
        print("🧪 Animal Type Classification System - Integration Tests")
        print("=" * 60)
        print()
        
        tests = [
            ("Backend Health", self.test_api_health),
            ("Root Endpoint", self.test_root_endpoint),
            ("Statistics Endpoint", self.test_statistics_endpoint),
            ("History Endpoint", self.test_history_endpoint),
            ("Image Upload & Classification", self.test_image_upload)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"Running: {test_name}")
            try:
                if test_func():
                    passed += 1
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                self.log_test(test_name, False, f"Test exception: {str(e)}")
            print()
        
        # Generate summary report
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print()
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! System is working correctly.")
        elif passed > 0:
            print("⚠️  Some tests failed. Check the logs above for details.")
        else:
            print("❌ ALL TESTS FAILED. Please check your setup.")
        
        print()
        print("📝 Detailed test results saved to test_results.json")
        
        # Save detailed results to file
        with open("test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        return passed == total

def main():
    """Main test function."""
    print("Starting ATC System Integration Tests...")
    print("Make sure the backend server is running before proceeding!")
    print("(Run: start_backend.bat)")
    
    input("Press Enter to continue when backend is ready...")
    
    tester = ATCTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 System is ready for use!")
        print("Next steps:")
        print("1. Start frontend: start_frontend.bat")
        print("2. Open browser: http://localhost:8501")
        print("3. Upload animal images for classification")
    else:
        print("\n🔧 Please check the troubleshooting guide for solutions.")
        print("See: TROUBLESHOOTING.md")
    
    return success

if __name__ == "__main__":
    main()