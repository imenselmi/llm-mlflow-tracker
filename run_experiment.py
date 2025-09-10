import mlflow
import requests
import json
import os

# Set your MLflow experiment name
import mlflow


# Then set your experiment
mlflow.set_experiment("LLM_Prompt_Experiments")  # or "LLM_Deployment_Monitoring"


# Choose the model to test
MODEL_PORTS = {
    "Qwen3-32B": 8015,
    "gemma-3-27b-it": 8018,
    "Qwen3-30B-A3B": 8017,
    "ensemble": 8014
}

def run_prompt(model_name, prompt_file):
    with open(prompt_file, "r") as f:
        prompt = f.read().strip()
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    port = MODEL_PORTS[model_name]
    response = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload)
    print("Raw response:", response.text)  # Debug
    
    completion_data = response.json()
    
    # Safely extract completion
    if "choices" in completion_data:
        completion = completion_data["choices"][0]["message"]["content"]
    elif "result" in completion_data:  # for DeepSeek or custom API
        completion = completion_data["result"]
    else:
        completion = f"Error: Unexpected response {completion_data}"

    # Log to MLflow
    with mlflow.start_run(run_name=f"{model_name}_{os.path.basename(prompt_file)}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("prompt_file", prompt_file)
        mlflow.log_metric("completion_length", len(completion))
        mlflow.log_text(completion, "completion.txt")
        print(f"Logged run for {model_name} with prompt {prompt_file}")


if __name__ == "__main__":
    for model in MODEL_PORTS.keys():
        for prompt_file in os.listdir("prompts"):
            if prompt_file.endswith(".txt"):
                run_prompt(model, os.path.join("prompts", prompt_file))
