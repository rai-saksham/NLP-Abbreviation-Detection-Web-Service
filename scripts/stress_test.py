from locust import HttpUser, task, constant
import json

class QuickstartUser(HttpUser):
    wait_time = constant(0)
    host = "http://localhost:8000"  # Update this to match your server's address if needed

    @task
    def test_post_method(self):
        model_endpoint = '/predict'
        text_request = {
            "text": "Abbreviations: GEMS, Global Enteric Multicenter Study; VIP, ventilated improved pit."
        }

        response = self.client.post(model_endpoint, json=text_request)
        
        try:
            response_json = response.json()
            print("Response JSON:")
            for item in response_json['result']:
                print(f"Label: {item['label']}, Word: {item['word']}, Score: {item['score']:.4f}")
        except json.JSONDecodeError as e:
            print(f"Error decoding response: {e}")
            print(f"Raw response text: {response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Raw response text: {response.text}")

