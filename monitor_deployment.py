import mlflow
import psutil
import requests
import time

MODEL_PORTS = {
    "Qwen3-32B": 8015,
    "gemma-3-27b-it": 8018,
    "Qwen3-30B-A3B": 8017,
    "ensemble": 8014
}

mlflow.set_experiment("LLM_Deployment_Monitoring")

def monitor(model_name, interval=60):
    port = MODEL_PORTS[model_name]
    while True:
        start = time.time()
        try:
            response = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                json={"model": model_name, "messages":[{"role":"user","content":"Ping"}]}
            )
            latency = time.time() - start
        except Exception:
            latency = None
        
        with mlflow.start_run(run_name=f"{model_name}_monitor"):
            mlflow.log_metric("cpu_usage", psutil.cpu_percent())
            mlflow.log_metric("memory_usage", psutil.virtual_memory().percent)
            if latency:
                mlflow.log_metric("latency", latency)
            print(f"{model_name}: CPU={psutil.cpu_percent()}%, Mem={psutil.virtual_memory().percent}%, Latency={latency}")
        
        time.sleep(interval)

if __name__ == "__main__":
    import threading
    for model in MODEL_PORTS.keys():
        t = threading.Thread(target=monitor, args=(model,))
        t.start()
