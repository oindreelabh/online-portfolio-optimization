#!/bin/bash

export AIRFLOW_HOME=$(pwd)/airflow

# Stop scheduler
if [ -f "$AIRFLOW_HOME/scheduler.pid" ]; then
    kill $(cat $AIRFLOW_HOME/scheduler.pid) && rm $AIRFLOW_HOME/scheduler.pid
    echo "Stopped Airflow scheduler."
else
    echo "No scheduler.pid file found."
fi

# Stop API server
if [ -f "$AIRFLOW_HOME/api-server.pid" ]; then
    kill $(cat $AIRFLOW_HOME/api-server.pid) && rm $AIRFLOW_HOME/api-server.pid
    echo "Stopped Airflow API server."
else
    echo "No api-server.pid file found."
fi

# Stop dag-processor
if [ -f "$AIRFLOW_HOME/dag-processor.pid" ]; then
    kill $(cat $AIRFLOW_HOME/dag-processor.pid) && rm $AIRFLOW_HOME/dag-processor.pid
    echo "Stopped Airflow dag-processor."
else
    echo "No dag-processor.pid file found."
fi

