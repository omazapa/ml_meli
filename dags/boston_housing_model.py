from datetime import datetime
import os
import numpy as np
import pandas as pd

from airflow.sdk import dag, task
from airflow.sdk import Param

import mlflow
import mlflow.keras
from tensorflow import keras
from tensorflow.keras import layers
from mlflow.models.signature import infer_signature

# from boston_housing_assets import data_processing
import kagglehub
import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from dvc.repo import Repo
from airflow.operators.python import get_current_context
import mlflow.data


# ====== Config ======
BASE = "/opt/airflow/data/boston_housing"

EXPERIMENT_NAME = "boston_housing_analysis"
TARGET_COL = "MEDV"  # target column name in the CSVs

TENSORBOARD_LOGS_DIR = "/opt/airflow/logs/tensorboard/boston_housing"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # fuerza CPU


# Define a custom Keras callback to log metrics to MLflow at the end of each epoch.
class MlflowEpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in ("loss", "mae", "val_loss", "val_mae"):
            v = logs.get(k)
            if v is not None and np.isfinite(v):
                mlflow.log_metric(k, float(v), step=epoch)


# Define the Keras TensorBoard callback.
logdir = TENSORBOARD_LOGS_DIR + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


@dag(
    dag_id="boston_housing_workflow",
    start_date=datetime(2025, 10, 1),
    schedule=[],
    catchup=False,
    tags=["ml", "keras", "assets"],
    # Hyperparameters of the model training
    params={
        # Architecture parameters
        "hidden_units": Param(64, type="integer", minimum=1, description="Neuronas en capa oculta"),
        "dropout_rate": Param(0.1, type="number", minimum=0, maximum=1, description="Tasa de dropout"),
        # Training parameters
        "learning_rate": Param(0.001, type="number", minimum=1e-6, description="Tasa de aprendizaje"),
        "batch_size": Param(
            32,
            type="integer",
            minimum=1,
            description="Batch size for training",
        ),
        "epochs": Param(
            10,
            type="integer",
            minimum=1,
            description="Number of epochs for training",
        ),
    },
)
def train_boston_from_preprocessing_asset():
    """DAG to train Boston Housing model using data from preprocessing asset."""

    @task
    def data_preprocessing() -> dict:
        filename = "HousingData.csv"
        path = os.path.join(kagglehub.dataset_download("altavish/boston-housing-dataset"), filename)
        print(f"📦 Dataset descargado: {path}")
        df = pd.read_csv(path)

        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        train, test = train_test_split(df_imputed, test_size=0.30, random_state=42, shuffle=True)
        test, val = train_test_split(test, test_size=0.5, random_state=42, shuffle=True)

        ctx = get_current_context()
        run_id = ctx["dag_run"].run_id

        base_dir = "/opt/airflow/data/boston_housing/" + run_id
        os.makedirs(base_dir, exist_ok=True)
        train_path = os.path.join(base_dir, "train.csv")
        val_path = os.path.join(base_dir, "validation.csv")
        test_path = os.path.join(base_dir, "test.csv")

        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)

        # Push data to DVC remote (Minio)
        DVC_BUCKET = os.getenv("DVC_BUCKET")
        MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        # Configure DVC remote
        repo = Repo("./")
        repo.config["remote"]["minio"] = {
            "url": f"s3://{DVC_BUCKET}",
            "endpointurl": MLFLOW_S3_ENDPOINT_URL,
            "access_key_id": AWS_ACCESS_KEY_ID,
            "secret_access_key": AWS_SECRET_ACCESS_KEY,
        }
        # Add files to DVC
        # Automatically stages the files for commit
        # Perform dvc commit automatically
        repo.add([train_path, val_path, test_path])
        # Push to remote
        repo.push(remote="minio")
        print(f"✅ Data split: train={len(train)}, val={len(val)}, test={len(test)}")

        return {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "run_id": run_id,
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
        }

    @task
    def data_load_and_prepare(data: dict) -> dict:
        train_path = data["train_path"]
        val_path = data["val_path"]
        test_path = data["test_path"]

        # Check that CSVs exist
        if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
            raise FileNotFoundError(f"Missing dataset: Wating for \n{train_path}\n{val_path}\n{test_path}")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # returning data to run the model training task
        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "run_id": data["run_id"],
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
        }

    @task
    def train_with_mlflow(data: dict, **kwargs):
        """Train model using MLflow tracking."""
        params = kwargs["params"]

        #  Split features/target
        def xy(d):
            X = d.drop(columns=[TARGET_COL]).to_numpy(dtype=np.float32)
            y = d[TARGET_COL].values
            return X, y

        X_train, y_train = xy(data["train_df"])
        X_val, y_val = xy(data["val_df"])
        X_test, y_test = xy(data["test_df"])

        meta = {
            "n_features": int(X_train.shape[1]),
            "target": TARGET_COL,
            "sizes": {
                "train": int(len(X_train)),
                "validation": int(len(X_val)),
                "test": int(len(X_test)),
            },
        }

        n_features = meta["n_features"]

        # Build model function
        def build_model():
            model = keras.Sequential(
                [
                    layers.Input(shape=(n_features,)),
                    layers.Dense(params["hidden_units"], activation="relu"),
                    layers.Dropout(params["dropout_rate"]),
                    layers.Dense(params["hidden_units"] // 2, activation="relu"),
                    layers.Dense(1),
                ]
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                loss="mse",
                metrics=["mae"],
            )
            return model

        mlflow.set_experiment(EXPERIMENT_NAME)
        train_dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
            data["train_df"],
            source=data["train_path"],
            name="boston-housing",
        )
        val_dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
            data["val_df"],
            source=data["val_path"],
            name="boston-housing",
        )

        test_dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
            data["test_df"],
            source=data["test_path"],
            name="boston-housing",
        )
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(val_dataset, context="validation")
            mlflow.log_input(test_dataset, context="test")
            model = build_model()
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                verbose=2,
                callbacks=[MlflowEpochLogger(), tensorboard_callback],
            )

            # Prediction on test set
            y_pred = model.predict(X_test)
            mse = np.mean(np.square(y_pred.flatten() - y_test))
            mae = np.mean(np.abs(y_pred.flatten() - y_test))

            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_mae", mae)

            # Infer model signature
            signature = infer_signature(X_test, y_pred)

            # Register the model with MLflow
            mlflow.keras.log_model(
                model,
                artifact_path="model",
                registered_model_name="boston_model",
                signature=signature,
            )
            model_uri = f"runs:/{run.info.run_id}/model"

            registered = mlflow.register_model(model_uri=model_uri, name="boston_model")

            return {
                "test_mse": float(mse),
                "test_mae": float(mae),
                "artifact_uri": mlflow.get_artifact_uri(),
                "registered_model_version": registered.version,
                "registered_model_id": registered.model_id,
            }

    return train_with_mlflow(data_load_and_prepare(data_preprocessing()))


boston_dag = train_boston_from_preprocessing_asset()
