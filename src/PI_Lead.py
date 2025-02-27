class PILeadController:
    def __init__(self, kp, ki, T1, T2, sample_time):
        """
        Initialize the PI-lead controller.

        Parameters:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            T1 (float): Time constant for the lead zero.
            T2 (float): Time constant for the lead pole (T1 > T2 for lead behavior).
            sample_time (float): Sampling period (dt).
        """
        self.kp = kp
        self.ki = ki
        self.T1 = T1
        self.T2 = T2
        self.dt = sample_time

        # Compute discrete lead filter coefficients using Tustin approximation.
        self.a0 = 2 * T2 + sample_time
        self.a1 = sample_time - 2 * T2
        self.b0 = 2 * T1 + sample_time
        self.b1 = sample_time - 2 * T1

        # Initialize PI integrator and filter state.
        self.integral = 0.0
        self.x_prev = 0.0  # previous PI output (filter input)
        self.y_prev = 0.0  # previous controller output (filter output)

    def reset(self):
        """Reset the controller state (integral and filter memory)."""
        self.integral = 0.0
        self.x_prev = 0.0
        self.y_prev = 0.0

    def update(self, error):
        """
        Update the controller with a new error measurement.

        Parameters:
            error (float): The difference between the setpoint and the process variable.

        Returns:
            float: The control signal.
        """
        # Update the PI controller part.
        self.integral += error * self.dt
        u_pi = self.kp * error + self.ki * self.integral

        # Pass the PI output through the lead filter.
        # Difference equation:
        # y[k] = (b0 * x[k] + b1 * x[k-1] - a1 * y[k-1]) / a0
        y = (self.b0 * u_pi + self.b1 * self.x_prev - self.a1 * self.y_prev) / self.a0

        # Update stored states for the next iteration.
        self.x_prev = u_pi
        self.y_prev = y

        return y

# Example usage:
if __name__ == "__main__":
    # Define controller parameters.
    kp = 1.0
    ki = 0.5
    T1 = 0.1   # Lead zero time constant
    T2 = 0.05  # Lead pole time constant (ensure T1 > T2 for lead action)
    dt = 0.01  # Sample time (10 ms)

    controller = PILeadController(kp, ki, T1, T2, dt)

    # Simulate controller updates with a constant error.
    error = 0.2
    for _ in range(100):
        control_signal = controller.update(error)
        print(control_signal)





