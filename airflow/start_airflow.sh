#!/bin/bash

# Kill any process using port 8793
kill -9 $(lsof -t -i:8793)
kill -9 $(lsof -t -i:8080)

# Set AIRFLOW_HOME to current directory + /airflow
export AIRFLOW_HOME=$(pwd)/airflow

echo "AIRFLOW_HOME set to: $AIRFLOW_HOME"

export PYTHONPATH=$(pwd)/src

echo "PYTHONPATH set to: $PYTHONPATH"

airflow webserver > $AIRFLOW_HOME/logs/webserver.log 2>&1 &
echo $! > $AIRFLOW_HOME/webserver.pid

# Create necessary folders
mkdir -p $AIRFLOW_HOME/dags
mkdir -p $AIRFLOW_HOME/logs
mkdir -p $AIRFLOW_HOME/plugins

# Initialize Airflow DB
airflow db migrate

echo "Airflow database migrated."

# Show dags_folder config
echo "dags_folder is set to:"
airflow config get-value core dags_folder

echo "any import errors will be shown below"
airflow dags list-import-errors

echo "dags are"
airflow dags list



# Start airflow dag-processor in background and save PID
# This helps with DAG parsing and processing (recommended for Airflow 3.x issues)
airflow dag-processor > $AIRFLOW_HOME/logs/dag-processor.log 2>&1 &
echo $! > $AIRFLOW_HOME/dag-processor.pid

# Start scheduler in background and save PID
airflow scheduler > $AIRFLOW_HOME/logs/scheduler.log 2>&1 &
echo $! > $AIRFLOW_HOME/scheduler.pid

# Start API server in background and save PID
airflow api-server --port 8080 > $AIRFLOW_HOME/logs/api-server.log 2>&1 &
echo $! > $AIRFLOW_HOME/api-server.pid

echo "Airflow dag-processor, scheduler and API server started."
echo "Access the UI at http://localhost:8080"

# --- OPTIONAL: Force DAG detection ---
# If your DAGs still do not appear, you can run this command manually in a separate terminal:
# airflow standalone
# This will force Airflow to reparse and register DAGs.
