#!/bin/bash

# Set AIRFLOW_HOME to current directory + /airflow
export AIRFLOW_HOME=$(pwd)/airflow

echo "AIRFLOW_HOME set to: $AIRFLOW_HOME"

# Create necessary folders
mkdir -p $AIRFLOW_HOME/dags
mkdir -p $AIRFLOW_HOME/logs
mkdir -p $AIRFLOW_HOME/plugins

# Initialize Airflow DB
airflow db migrate

# Start scheduler in background and save PID
airflow scheduler > $AIRFLOW_HOME/logs/scheduler.log 2>&1 &
echo $! > $AIRFLOW_HOME/scheduler.pid

# Start API server in background and save PID
airflow api-server --port 8090 > $AIRFLOW_HOME/logs/api-server.log 2>&1 &
echo $! > $AIRFLOW_HOME/api-server.pid

echo "Airflow scheduler and API server started."
echo "Access the UI at http://localhost:8090"
