import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from CSV
df = pd.read_csv("gpu_stats.csv")

# Convert 'timestamp' column to datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create output directory to store plots
output_dir = "gpu_plots"
os.makedirs(output_dir, exist_ok=True)

def plot_metric(metric, ylabel, title, filename):
    """Plot and save the metric over time for each GPU."""
    plt.figure(figsize=(10, 5))
    
    # Plot each GPU's data on the same graph
    for gpu_id in df['gpu_id'].unique():
        gpu_data = df[df['gpu_id'] == gpu_id]
        plt.plot(gpu_data['timestamp'], gpu_data[metric], label=f'GPU {gpu_id}')

    # Formatting the plot
    plt.xlabel('Timestamp')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()  # Close the plot to free memory

# Generate and save plots for different metrics
plot_metric('utilization', 'GPU Utilization (%)', 'GPU Utilization Over Time', 'gpu_utilization.png')
plot_metric('mem_usage', 'Memory Usage (%)', 'GPU Memory Usage Over Time', 'gpu_memory_usage.png')
plot_metric('temp', 'Temperature (°C)', 'GPU Temperature Over Time', 'gpu_temperature.png')
plot_metric('power', 'Power Draw (W)', 'GPU Power Draw Over Time', 'gpu_power_draw.png')

print(f"Plots saved to the '{output_dir}' directory.")
