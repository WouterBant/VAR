import numpy as np

class KalmanFilter2D:
    def __init__(self, initial_state, process_noise, measurement_noise):
        self.state = np.array(initial_state)  # [x, y, theta]
        self.covariance = np.eye(3)  # Initial uncertainty in state
        self.process_noise = np.array(process_noise)  # Process noise
        self.measurement_noise = np.array(measurement_noise)  # Measurement noise

    def predict(self, d_left, d_right, wheel_base):
        # Compute motion
        d_center = (d_left + d_right) / 2.0
        delta_theta = (d_right - d_left) / wheel_base
        theta = self.state[2]

        if abs(delta_theta) > 1e-6:
            # Robot is turning
            radius = d_center / delta_theta
            dx = radius * (np.sin(theta + delta_theta) - np.sin(theta))
            dy = radius * (-np.cos(theta + delta_theta) + np.cos(theta))
        else:
            # Robot is moving straight
            dx = d_center * np.cos(theta)
            dy = d_center * np.sin(theta)

        # Update state
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += delta_theta
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi  # Normalize theta

        # Jacobian of the motion model
        G = np.eye(3)
        G[0, 2] = -d_center * np.sin(theta)
        G[1, 2] = d_center * np.cos(theta)

        # Predict covariance
        self.covariance = G @ self.covariance @ G.T + self.process_noise

    def update(self, measurement):
        # Measurement model
        H = np.array([[1, 0, 0], [0, 1, 0]])  # We only measure x and y

        # Innovation
        z = np.array(measurement)
        z_pred = H @ self.state
        y = z - z_pred  # Residual

        # Innovation covariance
        S = H @ self.covariance @ H.T + self.measurement_noise

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state += K @ y
        self.covariance = (np.eye(3) - K @ H) @ self.covariance

    def get_state(self):
        return self.state
