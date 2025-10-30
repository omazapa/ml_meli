import requests
import datetime
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar un DAG en Airflow 3+ usando la API REST"
    )
    parser.add_argument(
        "--url", required=True, help="URL base de Airflow, ej: http://localhost:8080"
    )
    parser.add_argument("--username", required=True, help="Usuario de Airflow")
    parser.add_argument("--password", required=True, help="Contraseña de Airflow")
    parser.add_argument("--dag_id", required=True, help="ID del DAG a ejecutar")
    args = parser.parse_args()
    auth_url = f"{args.url}/auth/token"
    auth_data = {"username": args.username, "password": args.password}

    response = requests.post(auth_url, json=auth_data)
    # Obtener token de autenticación
    if response.status_code == 201:
        token = response.json()["access_token"]
        print("✅ Token obtenido correctamente:")
    else:
        print("❌ Error al autenticar:", response.status_code, response.text)
        exit(1)
    # Configurar encabezados con el token
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    utc_now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    
    # Payload para ejecutar el DAG
    payload = {
        "dag_run_id": f"manual__{utc_now}",
        "logical_date": utc_now,  # 👈 requerido en Airflow 3
        "conf": {"trigger_source": "python_api"},
    }

    # endpoint para ejecutar el DAG
    url = f"{args.url}/api/v2/dags/{args.dag_id}/dagRuns"

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code in (200, 201):
        print("✅ DAG ejecutado correctamente")
        print(response.json())
    else:
        print("❌ Error al ejecutar el DAG:", response.status_code, response.text)


if __name__ == "__main__":
    main()
