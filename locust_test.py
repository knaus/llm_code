from locust import HttpUser, task, between
import json
import os

# --- Configuration ---
# Replace with your actual GCP Project ID, Endpoint ID, and Location
# You can also set these as environment variables before running Locust
# e.g., export GCP_PROJECT_ID="your-project-id"
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
ENDPOINT_ID = os.environ.get("VERTEX_AI_ENDPOINT_ID", "your-vertex-ai-endpoint-id")
LOCATION = os.environ.get("GCP_LOCATION", "your-gcp-region") # e.g., us-central1

# Base URL for the Vertex AI endpoint prediction
# Format: https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict
# Locust will automatically prepend the host defined in the HttpUser class
# We will define the host below.
# The path for the predict method is /v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict

# Example prompt and parameters for Llama 3.1 inference
# Adjust these based on your expected workload and the specific Llama 3.1 model version deployed (8B, 70B)
# Note: Llama 3.1 405B is generally too large for a single L4 GPU.
# Ensure these parameters are compatible with your deployed model version.
PROMPT = "Tell me a fun fact about the universe."
GENERATION_PARAMETERS = {
    "maxOutputTokens": 128, # Adjust for desired average response length in tokens
    "temperature": 0.7,
    "topP": 0.95,
    "topK": 40
}

# --- Locust User Class ---
class VertexAIUser(HttpUser):
    # Define the host URL for the Vertex AI endpoint
    # This is the base part of the URL before the /v1/... path
    host = f"https://{LOCATION}-aiplatform.googleapis.com"

    # wait_time defines the time between tasks for a single user
    # This simulates think time between requests. Adjust based on your scenario.
    # between(1, 5) means wait between 1 and 5 seconds before starting the next task
    # For saturation testing, you might want a low wait_time or even constant(0)
    # to maximize the rate at which a single user *could* send requests if not limited by the server.
    # However, Locust's user count is the primary control for *concurrent* requests.
    wait_time = between(0.1, 0.5) # Example: wait between 0.1 and 0.5 seconds

    @task
    def predict_llama(self):
        # Define the request payload in the format expected by Vertex AI
        payload = {
            "instances": [
                {
                    # The key here might be 'content', 'prompt', or something else
                    # depending on the model and deployment method. 'content' is common.
                    "content": PROMPT
                }
            ],
            "parameters": GENERATION_PARAMETERS
        }

        # Define the full path for the POST request
        predict_path = f"/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"

        # Send the POST request
        # The 'name' parameter groups requests in the Locust UI.
        # Using a static string like "/predict" is recommended for aggregation.
        self.client.post(predict_path, json=payload, name="/predict")

# --- Instructions to Run ---
# 1. Save this code as a Python file (e.g., locustfile.py).
# 2. Replace the placeholder values for PROJECT_ID, ENDPOINT_ID, and LOCATION
#    directly in the script or by setting environment variables GCP_PROJECT_ID,
#    VERTEX_AI_ENDPOINT_ID, and GCP_LOCATION before running Locust.
# 3. Adjust PROMPT and GENERATION_PARAMETERS to match your expected workload.
#    Ensure 'maxOutputTokens' is set to a realistic value for your use case.
# 4. Ensure you have authenticated to Google Cloud and have permissions to invoke
#    the Vertex AI endpoint from where you are running Locust.
#    (e.g., by running `gcloud auth application-default login` or using a service account).
# 5. Install Locust: pip install locust google-cloud-aiplatform google-auth
#    (google-cloud-aiplatform and google-auth might be helpful for local testing setup,
#     though Locust's HttpUser directly makes HTTP requests).
# 6. Run Locust from your terminal in the same directory as the locustfile.py:
#    locust -f locustfile.py
# 7. Open your web browser to http://localhost:8089 (or the address shown in the terminal).
# 8. In the Locust UI:
#    - Leave the Host field blank (it's set in the script).
#    - Enter the desired number of "Users" (this controls the number of concurrent requests).
#    - Enter the "Hatch rate" (users to start per second).
# 9. Click "Start swarming".

# --- Measuring Saturation ---
# While the Locust test is running, simultaneously monitor the metrics for your
# Vertex AI Endpoint in the Google Cloud Console:
# - Go to Vertex AI -> Endpoints -> Select your endpoint -> "View endpoint metrics".
# - Pay close attention to the "GPU Utilization" metric.

# To find the saturation point:
# - Start with a low number of users in Locust (e.g., 10 or 20).
# - Observe the "Requests per second" in the Locust UI and the "GPU Utilization"
#   in Google Cloud Monitoring.
# - Gradually increase the number of users in the Locust UI (e.g., by 10 or 20 each time).
# - Look for the point where:
#   - The "Requests per second" in the Locust UI stops increasing significantly or plateaus.
#   - The "GPU Utilization" in Google Cloud Monitoring consistently reaches a high
#     percentage (e.g., 95-100%).
# - The number of concurrent users in Locust when you observe this plateau and high
#   GPU utilization is an estimate of the number of requests needed to saturate the GPU
#   for your specific model, prompt, and parameters.
# - Also, observe the "Average response time" and "95% percentile" latency in Locust.
#   These will likely increase sharply once the GPU is saturated.
