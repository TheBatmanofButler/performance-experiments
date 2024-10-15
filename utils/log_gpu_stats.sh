#!/bin/bash

# Configuration
LOG_FILE="gpu_stats.csv"
INTERVAL=5  # Log every 5 seconds

# Initialize CSV with headers
echo "timestamp,gpu_id,utilization,mem_usage,temp,power" > $LOG_FILE

# Trap SIGINT (Ctrl+C) to exit gracefully
trap "echo 'Script stopped by user'; exit" SIGINT

echo "Starting GPU logging. Press Ctrl+C to stop."

# Infinite loop to log the data
while true; do
    # Capture current timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    # Run nvidia-smi command and append to the log file
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
               --format=csv,noheader,nounits | while read line; do
        echo "$timestamp,$line" >> $LOG_FILE
    done

    # Wait for the specified interval
    sleep $INTERVAL
done
