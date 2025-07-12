import requests
import json
import time
from typing import Dict, List

class RecommendationTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_endpoint(self, user_id: str, payload: Dict, test_name: str):
        """Test a single recommendation request"""
        print(f"\n🧪 Testing: {test_name}")
        print(f"User ID: {user_id}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/recommendations/{user_id}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response_time = time.time() - start_time
            
            print(f"📊 Status Code: {response.status_code}")
            print(f"⏱️  Response Time: {response_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                self._validate_response_structure(result)
                self._print_recommendations(result)
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
            
            self.test_results.append({
                "test_name": test_name,
                "user_id": user_id,
                "payload": payload,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200
            })
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            self.test_results.append({
                "test_name": test_name,
                "user_id": user_id,
                "payload": payload,
                "error": str(e),
                "success": False
            })
    
    def _validate_response_structure(self, response: Dict):
        """Validate the response has expected structure"""
        required_fields = [
            "recommended_methods",
            "effectiveness_score", 
            "engagement_probability",
            "model_version"
        ]
        
        for field in required_fields:
            if field not in response:
                print(f"⚠️  Missing field: {field}")
            else:
                print(f"✅ Found field: {field}")
        
        # Validate recommended_methods structure
        if "recommended_methods" in response:
            methods = response["recommended_methods"]
            if isinstance(methods, list) and len(methods) > 0:
                first_method = methods[0]
                method_fields = ["method", "confidence", "reason"]
                for field in method_fields:
                    if field not in first_method:
                        print(f"⚠️  Missing method field: {field}")
    
    def _print_recommendations(self, response: Dict):
        """Pretty print the recommendations"""
        print("\n📋 Recommendations:")
        
        if "recommended_methods" in response:
            for i, method in enumerate(response["recommended_methods"], 1):
                print(f"  {i}. {method.get('method', 'Unknown')}")
                print(f"     Confidence: {method.get('confidence', 0):.2f}")
                print(f"     Reason: {method.get('reason', 'No reason provided')}")
                if 'estimated_duration' in method:
                    print(f"     Duration: {method['estimated_duration']} min")
                if 'required_equipment' in method:
                    print(f"     Equipment: {method['required_equipment']}")
        
        print(f"\n📈 Scores:")
        print(f"  Effectiveness: {response.get('effectiveness_score', 'N/A')}")
        print(f"  Engagement: {response.get('engagement_probability', 'N/A')}")
        print(f"  Model Version: {response.get('model_version', 'N/A')}")
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting Recommendation Endpoint Tests")
        print("=" * 50)
        
        # Test cases
        test_cases = [
            {
                "name": "Empty Request",
                "user_id": "test_user_001",
                "payload": {}
            },
            {
                "name": "Home - Quick Session",
                "user_id": "test_user_002", 
                "payload": {
                    "currentLocation": "home",
                    "availableTime": 10,
                    "availableEquipment": ["yoga_mat"]
                }
            },
            {
                "name": "Gym - Full Session",
                "user_id": "test_user_003",
                "payload": {
                    "currentLocation": "gym",
                    "availableTime": 45,
                    "availableEquipment": ["foam_roller", "massage_gun", "sauna", "ice_bath"]
                }
            },
            {
                "name": "Office - No Equipment",
                "user_id": "test_user_004",
                "payload": {
                    "currentLocation": "office", 
                    "availableTime": 15,
                    "availableEquipment": []
                }
            },
            {
                "name": "Hotel - Limited Options",
                "user_id": "test_user_005",
                "payload": {
                    "currentLocation": "hotel",
                    "availableTime": 20,
                    "availableEquipment": ["towel", "wall"]
                }
            },
            {
                "name": "Edge Case - Very Short Time",
                "user_id": "test_user_006",
                "payload": {
                    "currentLocation": "anywhere",
                    "availableTime": 3,
                    "availableEquipment": []
                }
            },
            {
                "name": "Edge Case - Very Long Time",
                "user_id": "test_user_007",
                "payload": {
                    "currentLocation": "gym",
                    "availableTime": 90,
                    "availableEquipment": ["Foam Roller", "Massage Gun", "Sauna", "Ice Bath", "Yoga Mat"]
                }
            },
            {
                "name": "Same User - Different Context",
                "user_id": "test_user_002",  # Reuse user to test personalization
                "payload": {
                    "currentLocation": "gym",
                    "availableTime": 30,
                    "availableEquipment": ["Ice Bath", "Normatec"]
                }
            },
                        {
                "name": "oogie boogie",
                "user_id": "test_user_002",  # Reuse user to test personalization
                "payload": {
                    "currentLocation": "gym",
                    "availableTime": 30,
                    "availableEquipment": ["Hypervolt Gun", "Hot Pad"]
                }
            }
        ]
        
        # Run each test
        for test_case in test_cases:
            self.test_endpoint(
                test_case["user_id"],
                test_case["payload"],
                test_case["name"]
            )
            time.sleep(1)  # Brief pause between tests
        
        # Print summary
        self._print_test_summary()
    
    def _print_test_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 50)
        print("🧪 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get("success", False))
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Show failed tests
        failed_tests = [result for result in self.test_results if not result.get("success", False)]
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"  - {test['test_name']}: {test.get('error', 'HTTP Error')}")
        
        # Show response times
        response_times = [result.get("response_time", 0) for result in self.test_results if result.get("response_time")]
        if response_times:
            print(f"\n⏱️  Response Times:")
            print(f"  Average: {sum(response_times)/len(response_times):.2f}s")
            print(f"  Min: {min(response_times):.2f}s")
            print(f"  Max: {max(response_times):.2f}s")

if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"⚠️  Server responded with {response.status_code}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("Make sure to run: python RAG_server.py")
        exit(1)
    
    # Run tests
    tester = RecommendationTester()
    tester.run_all_tests()


# class LocalMLflowModels:
#     """Local MLflow model management"""
    
#     def __init__(self):
#         # Set local MLflow tracking
#         # mlflow.set_tracking_uri(f"file://{LOCAL_MLRUNS_PATH}")
#         mlflow.set_tracking_uri("./mlruns")
#         mlflow.set_experiment("recovery_recommendations")
        
#         self.models = {}
#         self.encoders = {}
#         self.load_models()
    
#     def load_models(self):
#         """Load saved models"""
#         try:
#             if (LOCAL_MODELS_PATH / "effectiveness_model.pkl").exists():
#                 with open(LOCAL_MODELS_PATH / "effectiveness_model.pkl", 'rb') as f:
#                     self.models['effectiveness'] = pickle.load(f)
                
#             if (LOCAL_MODELS_PATH / "engagement_model.pkl").exists():
#                 with open(LOCAL_MODELS_PATH / "engagement_model.pkl", 'rb') as f:
#                     self.models['engagement'] = pickle.load(f)
                
#             if (LOCAL_MODELS_PATH / "encoders.pkl").exists():
#                 with open(LOCAL_MODELS_PATH / "encoders.pkl", 'rb') as f:
#                     self.encoders = pickle.load(f)
                    
#             logger.info(f"Loaded {len(self.models)} models")
            
#         except Exception as e:
#             logger.error(f"Error loading models: {e}")
    
#     def predict_recommendations(self, user_features: Dict) -> Dict:
#         """Generate recommendations using local models"""
#         if not self.models:
#             # Return default recommendations
#             return self.get_default_recommendations()
        
#         try:
#             # Prepare features
#             feature_df = pd.DataFrame([user_features])
            
#             # Encode categorical features
#             if 'fitness_level' in self.encoders:
#                 feature_df['fitness_level_encoded'] = self.encoders['fitness_level'].transform(
#                     [user_features.get('fitness_level', 'beginner')]
#                 )
#             else:
#                 feature_df['fitness_level_encoded'] = 0
                
#             if 'age_group' in self.encoders:
#                 feature_df['age_group_encoded'] = self.encoders['age_group'].transform(
#                     [user_features.get('age_group', 'unknown')]
#                 )
#             else:
#                 feature_df['age_group_encoded'] = 0
            
#             # Select features
#             feature_cols = [
#                 'total_interactions', 'unique_sessions', 'avg_response_length',
#                 'total_sessions', 'avg_session_duration', 'avg_satisfaction',
#                 'avg_methods_completed', 'completion_efficiency',
#                 'avg_method_effectiveness', 'fitness_level_encoded', 'age_group_encoded'
#             ]
            
#             # Fill missing values
#             for col in feature_cols:
#                 if col not in feature_df.columns:
#                     feature_df[col] = 0
            
#             X = feature_df[feature_cols]
            
#             # Predict
#             effectiveness_score = float(self.models['effectiveness'].predict(X)[0])
#             engagement_prob = float(self.models['engagement'].predict_proba(X)[0, 1])
            
#             # Generate recommendations based on scores
#             recommendations = self.generate_method_recommendations(
#                 effectiveness_score, engagement_prob
#             )
            
#             return {
#                 "recommended_methods": recommendations,
#                 "effectiveness_score": effectiveness_score,
#                 "engagement_probability": engagement_prob,
#                 "model_version": "local_v1",
#                 "user_profile": {
#                     "fitness_level": user_features.get('fitness_level', 'unknown'),
#                     "avg_satisfaction": user_features.get('avg_satisfaction', 3.0)
#                 }
#             }
            
#         except Exception as e:
#             logger.error(f"Error predicting: {e}")
#             return self.get_default_recommendations()
    
#     def generate_method_recommendations(self, effectiveness_score: float, 
#                                       engagement_prob: float) -> List[Dict]:
#         """Generate method recommendations based on scores"""
#         if effectiveness_score >= 4.0 and engagement_prob >= 0.7:
#             return [
#                 {"method": "Cold Water Immersion", "confidence": 0.9, "reason": "High effectiveness user"},
#                 {"method": "Foam Rolling - Legs", "confidence": 0.85, "reason": "Proven results"},
#                 {"method": "Deep Breathing", "confidence": 0.8, "reason": "High completion rate"}
#             ]
#         elif effectiveness_score >= 3.5:
#             return [
#                 {"method": "Gentle Stretching", "confidence": 0.75, "reason": "Moderate effectiveness"},
#                 {"method": "Progressive Muscle Relaxation", "confidence": 0.7, "reason": "Good for progression"},
#                 {"method": "Deep Breathing", "confidence": 0.8, "reason": "Always beneficial"}
#             ]
#         else:
#             return [
#                 {"method": "Deep Breathing", "confidence": 0.8, "reason": "Easy to start"},
#                 {"method": "Legs Up The Wall", "confidence": 0.75, "reason": "Passive recovery"},
#                 {"method": "Tennis Ball Foot Massage", "confidence": 0.7, "reason": "Quick and simple"}
#             ]
    
#     def get_default_recommendations(self) -> Dict:
#         """Default recommendations when models aren't available"""
#         return {
#             "recommended_methods": [
#                 {"method": "Deep Breathing", "confidence": 0.8, "reason": "Universal benefit"},
#                 {"method": "Gentle Stretching", "confidence": 0.75, "reason": "Safe for all levels"},
#                 {"method": "Foam Rolling - Legs", "confidence": 0.7, "reason": "Popular choice"}
#             ],
#             "effectiveness_score": 3.5,
#             "engagement_probability": 0.6,
#             "model_version": "default",
#             "user_profile": {"fitness_level": "unknown"}
#         }