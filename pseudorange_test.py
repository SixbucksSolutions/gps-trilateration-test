import numpy
import numpy.typing
import scipy.optimize

# ==========================================
# STEP 1: Speed of Light Constant
# ==========================================
# Crucial for converting the 4th dimension (meters) back into time (seconds).
SPEED_OF_LIGHT: float = 299_792_458.0

# ==========================================
# STEP 2: Input Data Provided by the Satellites
# ==========================================
# 3D ECEF positions of 4 visible GPS satellites (units: meters)
# These represent known coordinates extracted from the satellite ephemeris data.
satellite_positions: numpy.typing.NDArray = numpy.array([
    [ 15_600_000.0,  16_500_000.0,  15_200_000.0],  # Sat 1
    [ 22_300_000.0,  -2_300_000.0,  13_800_000.0],  # Sat 2
    [  1_900_000.0, -21_500_000.0,  15_700_000.0],  # Sat 3
    [ 16_200_000.0, -13_200_000.0, -17_200_000.0]   # Sat 4
])

# Raw pseudoranges measured by the receiver hardware (units: meters).
# These numbers are inherently corrupted by the unknown receiver clock bias.
measured_pseudoranges: numpy.typing.NDArray = numpy.array([
    23_549_221.73,  # Distance to Sat 1 + (c * bias)
    21_503_378.13,  # Distance to Sat 2 + (c * bias)
    20_531_644.02,  # Distance to Sat 3 + (c * bias)
    24_467_005.10   # Distance to Sat 4 + (c * bias)
])


# ==========================================
# STEP 3: Define the Mathematical Residuals
# ==========================================
def gps_residuals(
        state_guess: numpy.typing.NDArray,
        sat_pos: numpy.typing.NDArray,
        pseudoranges: numpy.typing.NDArray) -> numpy.typing.NDArray:
    """
    Calculates how well our current guess fits the raw physics equations.
    Goal of the optimizer is to drive all 4 output values as close to 0 as possible.
    """
    # state_guess contains 4 elements: [X, Y, Z, clock_bias_in_meters]
    estimated_pos = state_guess[0:3]
    estimated_bias_meters = state_guess[3]

    residuals: numpy.typing.NDArray = numpy.zeros(4)
    for i in range(4):
        # 1. Compute true geometric range for the current guess
        calculated_dist = numpy.linalg.norm(sat_pos[i] - estimated_pos)

        # 2. Add the guessed clock bias (converted to meters)
        modeled_pseudorange = calculated_dist + estimated_bias_meters

        # 3. Compute the discrepancy (residual) against actual data
        residuals[i] = modeled_pseudorange - pseudoranges[i]

    return residuals


# ==========================================
# STEP 4: Execution from Absolute Zero
# ==========================================
# The cold start: we assume the receiver is at the exact center of the Earth
# with a perfectly synchronized clock. No hidden data is referenced.
cold_start_guess: numpy.typing.NDArray = numpy.array([0.0, 0.0, 0.0, 0.0])

# Solve the non-linear system of equations
solver_output = scipy.optimize.least_squares(
    gps_residuals,
    cold_start_guess,
    args=(satellite_positions, measured_pseudoranges),
    method='lm' # Levenberg-Marquardt algorithm is ideal for tracking/least-squares
)

# Extract computed parameters
estimated_state = solver_output.x
computed_pos: numpy.typing.NDArray = estimated_state[0:3]
computed_bias_seconds: float = estimated_state[3] / SPEED_OF_LIGHT

# ==========================================
# STEP 5: Print Final Output Metrics
# ==========================================
print("==================================================")
print("             GPS PVT ENGINE OUTPUT                ")
print("==================================================")
print(f"Computed Receiver Position (ECEF Meters):")
print(f"  X: {computed_pos[0]:14,.02f}")
print(f"  Y: {computed_pos[1]:14,.02f}")
print(f"  Z: {computed_pos[2]:14,.02f}")
print("--------------------------------------------------")
print(f"Computed Receiver Clock Offset:")
print(f"  In Meters:  {estimated_state[3]:14,.02f} m")
print(f"  In Seconds:          {computed_bias_seconds:9,.06f} s")
print("==================================================")
print(f"Optimizer Success Status: {solver_output.success}")
print(f"Final Residual Cost:      {solver_output.cost:.4e}")