import numpy as np
import pandas as pd
import json
from pathlib import Path

class DroneFlightSimulator:
    def __init__(self, dt=0.01, duration=100, flight_id=1):
        self.dt = dt
        self.duration = duration
        self.num_steps = int(duration / dt)
        self.flight_id = flight_id
        self.g = 9.81
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.attitude = np.array([0.0, 0.0, 0.0])
        self.angular_rate = np.array([0.0, 0.0, 0.0])
        self.timestamps = []
        self.imu_data = []
        self.gps_data = []
        self.ground_truth = []
    
    def simulate_step(self, t):
        radius = 5.0
        omega = 2 * np.pi / 30
        desired_x = radius * np.cos(omega * t)
        desired_y = radius * np.sin(omega * t)
        desired_z = min(t * 0.5, 10.0)
        self.position = np.array([desired_x, desired_y, desired_z])
        next_t = t + self.dt
        next_x = radius * np.cos(omega * next_t)
        next_y = radius * np.sin(omega * next_t)
        self.velocity[0] = (next_x - desired_x) / self.dt
        self.velocity[1] = (next_y - desired_y) / self.dt
        self.velocity[2] = 0.5 if t < 20 else 0
    
    def generate_imu_measurement(self, t, add_noise=True):
        accel = np.array([0.1, 0.05, 9.81])
        if add_noise:
            accel += np.random.randn(3) * 0.01
        gyro = np.array([0.01, 0.01, 0.0])
        if add_noise:
            gyro += np.random.randn(3) * 0.001
        return accel, gyro
    
    def generate_gps_measurement(self, t, gps_available=True, add_noise=True):
        if not gps_available:
            return None
        gps_pos = self.position.copy()
        if add_noise:
            gps_pos += np.random.randn(3) * np.array([1.0, 1.0, 2.0])
        return gps_pos
    
    def create_gps_dropout_schedule(self):
        gps_available = np.ones(self.num_steps, dtype=bool)
        dropout_events = [(30, 35), (55, 70), (85, 88)]
        for start_s, end_s in dropout_events:
            start_idx = int(start_s / self.dt)
            end_idx = int(end_s / self.dt)
            gps_available[start_idx:end_idx] = False
        return gps_available
    
    def simulate_flight(self):
        gps_available = self.create_gps_dropout_schedule()
        for step in range(self.num_steps):
            t = step * self.dt
            self.simulate_step(t)
            accel, gyro = self.generate_imu_measurement(t)
            gps_pos = self.generate_gps_measurement(t, gps_available[step])
            self.timestamps.append(t)
            self.imu_data.append(np.concatenate([accel, gyro]))
            self.gps_data.append(gps_pos if gps_pos is not None else np.array([np.nan, np.nan, np.nan]))
            self.ground_truth.append(np.concatenate([self.position, self.velocity, self.attitude]))
        return {
            'timestamps': np.array(self.timestamps),
            'imu': np.array(self.imu_data),
            'gps': np.array(self.gps_data),
            'ground_truth': np.array(self.ground_truth),
        }
    
    def save_to_files(self, output_dir='data'):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        flight_dir = Path(output_dir) / f"flight_{self.flight_id:03d}"
        flight_dir.mkdir(exist_ok=True)
        data = self.simulate_flight()
        df_imu = pd.DataFrame(data['imu'], columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        df_imu['timestamp'] = data['timestamps']
        df_imu.to_csv(flight_dir / 'imu_raw.csv', index=False)
        df_gps = pd.DataFrame(data['gps'], columns=['px', 'py', 'pz'])
        df_gps['timestamp'] = data['timestamps']
        df_gps.to_csv(flight_dir / 'gps_raw.csv', index=False)
        df_gt = pd.DataFrame(data['ground_truth'], columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw'])
        df_gt['timestamp'] = data['timestamps']
        df_gt.to_csv(flight_dir / 'ground_truth.csv', index=False)

if __name__ == "__main__":
    print("Generating 50 synthetic flights...")
    for i in range(1, 51):
        sim = DroneFlightSimulator(flight_id=i)
        sim.save_to_files()
        if i % 10 == 0:
            print(f"  {i}/50 complete")
    print("✓ Data generation done!")
