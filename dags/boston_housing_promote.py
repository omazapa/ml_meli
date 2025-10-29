from datetime import datetime
from airflow.sdk import dag, task
from mlflow.tracking import MlflowClient
import requests
import os
import time

MODEL_NAME = "boston_model"

client = MlflowClient()
model_version: int | None = None
if model_version is None:
    try:
        # Get latest version
        versions = client.get_latest_versions(name=MODEL_NAME)
        if not versions:
            raise ValueError(f"No versions found for model '{MODEL_NAME}'")
        model_version = max(int(v.version) for v in versions)
        print(f"No version provided, defaulting to latest version: {model_version}")
    except Exception as e:
        print(f"WARNING: no model versions found for '{MODEL_NAME}': {e}")


@dag(
    dag_id="boston_housing_promote_model",
    start_date=datetime(2025, 10, 1),
    schedule=None,
    catchup=False,
    tags=["mlflow", "production"],
    params={"version": model_version},  # default param
)
def promote_model_dag():

    @task()
    def promote_model(**kwargs):
        """
        Promote a registered MLflow model version to Production stage.
        If no version is provided, defaults to the latest available version.
        """
        version = kwargs["params"].get("version")
        # If no version passed, get the latest version dynamically
        if version is None:
            print(f"WARNING:No version provided, skipping promotion.")
            raise ValueError("No version provided for promotion task")

        try:
            print(f"Promoting model '{MODEL_NAME}' version {version} to Production...")
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=int(version),
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"Model '{MODEL_NAME}' version {version} is now in Production")
        except Exception as e:
            print(f"ERROR: promoting model: {e}")
            raise
        return version

    @task()
    def wait_until_promoted(version: int):
        """
        Polls MLflow Model Registry until the model version is confirmed in 'Production' stage.
        """
        print(f"Waiting until model '{MODEL_NAME}' version {version} is in Production...")
        for i in range(10):  # retry 10 times (about ~1 min total)
            mv = client.get_model_version(name=MODEL_NAME, version=str(version))
            current_stage = mv.current_stage
            print(f"Attempt {i+1}: current stage = {current_stage}")
            if current_stage == "Production":
                print("✅ Model is now in Production.")
                return True
            time.sleep(6)  # wait 6 seconds before checking again
        raise TimeoutError(f"Model '{MODEL_NAME}' version {version} did not reach Production stage in time.")

    @task()
    def reload_model():
        """
        Calls the /reload endpoint of MLflow API to reload the model recently promoted to Production.
        """
        MLFLOWAPI_API_KEY = os.getenv("MLFLOWAPI_API_KEY")
        MLFLOWAPI_SERVER_URL = os.getenv("MLFLOWAPI_SERVER_URL")
        print("INFO: Calling MLflow API /reload endpoint to reload the model...")
        headers = {"Content-Type": "application/json", "X-API-KEY": MLFLOWAPI_API_KEY}
        try:
            response = requests.post(MLFLOWAPI_SERVER_URL + "/reload", headers=headers, timeout=30)
            if response.status_code == 200:
                print("Model reloaded successfully after promotion")
            else:
                print(f"Failed to reload model: {response.status_code} {response.text}")
                response.raise_for_status()
        except Exception as e:
            print(f"Error calling MLflow API /reload: {e}")
            raise

    wait_until_promoted(promote_model()) >> reload_model()


# Instantiate the DAG
dag_promote = promote_model_dag()
