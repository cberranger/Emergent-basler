#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Basler Camera Application
Tests all API endpoints with no physical cameras connected
"""

import requests
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

class BaslerCameraAPITester:
    def __init__(self, base_url="https://blaze-capture-ui.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.session = requests.Session()
        self.session.timeout = 10

    def log_test(self, name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            
        result = {
            "test_name": name,
            "success": success,
            "details": details,
            "response_data": response_data,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if details:
            print(f"    Details: {details}")
        if not success and response_data:
            print(f"    Response: {response_data}")
        print()

    def test_api_root(self) -> bool:
        """Test GET /api/ - Basic API info"""
        try:
            response = self.session.get(f"{self.api_url}/")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                expected_message = "Basler Camera Application API"
                if data.get("message") == expected_message:
                    self.log_test("API Root Endpoint", True, f"Status: {response.status_code}, Message: {data.get('message')}")
                else:
                    self.log_test("API Root Endpoint", False, f"Unexpected message: {data.get('message')}")
                    success = False
            else:
                self.log_test("API Root Endpoint", False, f"Status: {response.status_code}", response.text)
                
            return success
            
        except Exception as e:
            self.log_test("API Root Endpoint", False, f"Exception: {str(e)}")
            return False

    def test_get_cameras(self) -> bool:
        """Test GET /api/cameras - Get camera list"""
        try:
            response = self.session.get(f"{self.api_url}/cameras")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                if isinstance(data, list):
                    self.log_test("Get Cameras List", True, f"Status: {response.status_code}, Cameras found: {len(data)}", data)
                else:
                    self.log_test("Get Cameras List", False, f"Expected list, got: {type(data)}")
                    success = False
            else:
                self.log_test("Get Cameras List", False, f"Status: {response.status_code}", response.text)
                
            return success
            
        except Exception as e:
            self.log_test("Get Cameras List", False, f"Exception: {str(e)}")
            return False

    def test_system_status(self) -> bool:
        """Test GET /api/status - Get system status"""
        try:
            response = self.session.get(f"{self.api_url}/status")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                required_fields = ["timestamp", "total_cameras", "connected_cameras", "streaming_cameras", "camera_support", "cameras"]
                
                missing_fields = [field for field in required_fields if field not in data]
                if not missing_fields:
                    camera_support = data.get("camera_support")
                    total_cameras = data.get("total_cameras", 0)
                    self.log_test("System Status", True, 
                                f"Status: {response.status_code}, Camera Support: {camera_support}, Total Cameras: {total_cameras}", 
                                data)
                else:
                    self.log_test("System Status", False, f"Missing fields: {missing_fields}")
                    success = False
            else:
                self.log_test("System Status", False, f"Status: {response.status_code}", response.text)
                
            return success
            
        except Exception as e:
            self.log_test("System Status", False, f"Exception: {str(e)}")
            return False

    def test_camera_operations_no_cameras(self) -> bool:
        """Test camera operations when no cameras are connected"""
        fake_camera_id = "cam_0"
        
        # Test connect to non-existent camera
        try:
            response = self.session.post(f"{self.api_url}/cameras/{fake_camera_id}/connect")
            # Should fail with 400 since no cameras exist
            if response.status_code == 400:
                self.log_test("Connect Non-existent Camera", True, f"Correctly returned 400 for non-existent camera")
            else:
                self.log_test("Connect Non-existent Camera", False, f"Expected 400, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Connect Non-existent Camera", False, f"Exception: {str(e)}")

        # Test disconnect non-existent camera
        try:
            response = self.session.post(f"{self.api_url}/cameras/{fake_camera_id}/disconnect")
            # Should succeed (graceful handling)
            if response.status_code == 200:
                self.log_test("Disconnect Non-existent Camera", True, f"Gracefully handled disconnect")
            else:
                self.log_test("Disconnect Non-existent Camera", False, f"Expected 200, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Disconnect Non-existent Camera", False, f"Exception: {str(e)}")

        # Test configure non-existent camera
        try:
            config_data = {
                "exposure_time": 1000,
                "gain": 1.0,
                "frame_rate": 30.0
            }
            response = self.session.post(f"{self.api_url}/cameras/{fake_camera_id}/configure", json=config_data)
            # Should fail with 400
            if response.status_code == 400:
                self.log_test("Configure Non-existent Camera", True, f"Correctly returned 400 for configuration")
            else:
                self.log_test("Configure Non-existent Camera", False, f"Expected 400, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Configure Non-existent Camera", False, f"Exception: {str(e)}")

        # Test start streaming non-existent camera
        try:
            response = self.session.post(f"{self.api_url}/cameras/{fake_camera_id}/start-streaming")
            # Should fail with 400
            if response.status_code == 400:
                self.log_test("Start Streaming Non-existent Camera", True, f"Correctly returned 400 for streaming")
            else:
                self.log_test("Start Streaming Non-existent Camera", False, f"Expected 400, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Start Streaming Non-existent Camera", False, f"Exception: {str(e)}")

        # Test stop streaming non-existent camera
        try:
            response = self.session.post(f"{self.api_url}/cameras/{fake_camera_id}/stop-streaming")
            # Should succeed (graceful handling)
            if response.status_code == 200:
                self.log_test("Stop Streaming Non-existent Camera", True, f"Gracefully handled stop streaming")
            else:
                self.log_test("Stop Streaming Non-existent Camera", False, f"Expected 200, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Stop Streaming Non-existent Camera", False, f"Exception: {str(e)}")

        # Test get frame from non-existent camera
        try:
            response = self.session.get(f"{self.api_url}/cameras/{fake_camera_id}/frame")
            # Should fail with 404
            if response.status_code == 404:
                self.log_test("Get Frame Non-existent Camera", True, f"Correctly returned 404 for frame request")
            else:
                self.log_test("Get Frame Non-existent Camera", False, f"Expected 404, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Get Frame Non-existent Camera", False, f"Exception: {str(e)}")

        return True

    def test_save_configuration(self) -> bool:
        """Test POST /api/save-config - Set save configuration"""
        try:
            save_config = {
                "enabled": True,
                "base_directory": "/tmp/camera_captures",
                "create_timestamp_folder": True,
                "save_format": "png"
            }
            
            response = self.session.post(f"{self.api_url}/save-config", json=save_config)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                self.log_test("Save Configuration", True, f"Status: {response.status_code}, Message: {data.get('message')}")
            else:
                self.log_test("Save Configuration", False, f"Status: {response.status_code}", response.text)
                
            return success
            
        except Exception as e:
            self.log_test("Save Configuration", False, f"Exception: {str(e)}")
            return False

    def test_api_timeout_behavior(self) -> bool:
        """Test API timeout behavior"""
        try:
            # Test with very short timeout to see if API handles it gracefully
            short_session = requests.Session()
            short_session.timeout = 0.001  # 1ms timeout
            
            try:
                response = short_session.get(f"{self.api_url}/cameras")
                self.log_test("API Timeout Behavior", False, "Expected timeout but got response")
            except requests.exceptions.Timeout:
                self.log_test("API Timeout Behavior", True, "API correctly times out with short timeout")
            except Exception as e:
                self.log_test("API Timeout Behavior", True, f"API handled timeout gracefully: {type(e).__name__}")
                
            return True
            
        except Exception as e:
            self.log_test("API Timeout Behavior", False, f"Exception: {str(e)}")
            return False

    def test_invalid_endpoints(self) -> bool:
        """Test invalid endpoints return proper errors"""
        invalid_endpoints = [
            "/api/invalid",
            "/api/cameras/invalid/connect",
            "/api/nonexistent"
        ]
        
        for endpoint in invalid_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code == 404:
                    self.log_test(f"Invalid Endpoint {endpoint}", True, f"Correctly returned 404")
                else:
                    self.log_test(f"Invalid Endpoint {endpoint}", False, f"Expected 404, got {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"Invalid Endpoint {endpoint}", False, f"Exception: {str(e)}")
        
        return True

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all backend API tests"""
        print("🚀 Starting Basler Camera API Backend Tests")
        print(f"📡 Testing API at: {self.api_url}")
        print("=" * 60)
        
        # Core API tests
        self.test_api_root()
        self.test_get_cameras()
        self.test_system_status()
        
        # Camera operation tests (no physical cameras)
        self.test_camera_operations_no_cameras()
        
        # Configuration tests
        self.test_save_configuration()
        
        # Error handling tests
        self.test_api_timeout_behavior()
        self.test_invalid_endpoints()
        
        # Summary
        print("=" * 60)
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {self.tests_run}")
        print(f"   Passed: {self.tests_passed}")
        print(f"   Failed: {self.tests_run - self.tests_passed}")
        print(f"   Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        return {
            "total_tests": self.tests_run,
            "passed_tests": self.tests_passed,
            "failed_tests": self.tests_run - self.tests_passed,
            "success_rate": (self.tests_passed/self.tests_run)*100,
            "test_results": self.test_results,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main test execution"""
    tester = BaslerCameraAPITester()
    results = tester.run_all_tests()
    
    # Save results to file
    results_file = "/app/test_reports/backend_api_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Detailed results saved to: {results_file}")
    
    # Return appropriate exit code
    return 0 if results["failed_tests"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())