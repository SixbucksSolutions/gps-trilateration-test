import argparse

import numpy
import numpy.typing
import pymap3d
import scipy.optimize


# ==========================================
# STEP 1: Speed of Light Constant
# ==========================================
# Crucial for converting the 4th dimension (meters) back into time (seconds).
SPEED_OF_LIGHT: float = 299_792_458.0



def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Trilateration demo for GPS")
    parser.add_argument("local_clock_offset_seconds", type=float,
        help="Local clock offset from GPS time in seconds (-1.5 .. +1.5)")
    return parser.parse_args()


def _determine_measured_pseudoranges(args: argparse.Namespace,
                                     hidden_actual_receiver_ecef_pos: numpy.typing.NDArray,
                                     satellite_positions: numpy.typing.NDArray) -> numpy.typing.NDArray:

    clock_bias_seconds: float = args.local_clock_offset_seconds

    if not -1.5 <= clock_bias_seconds <= 1.5:
        raise ValueError(f"Clock bias must be between -1.5 and +1.5 seconds to let least squares find a solution")

    print()
    print(f"    Clock bias (seconds) : {clock_bias_seconds:21,.06f} s")
    clock_bias_meters: float = clock_bias_seconds * SPEED_OF_LIGHT
    print(f"    Clock bias (meters)  : {clock_bias_meters:21,.06f} m")

    # 3. Compute vector differences (subtract receiver position from each satellite)
    delta_vectors: numpy.typing.NDArray = satellite_positions - hidden_actual_receiver_ecef_pos

    # 4. Compute true geometric range (L2 Euclidean norm along the row axis)
    geometric_ranges: numpy.typing.NDArray = numpy.linalg.norm(delta_vectors, axis=1)

    # 5. Add receiver clock bias in meters to get pseudoranges
    raw_measured_pseudoranges: numpy.typing.NDArray = geometric_ranges + clock_bias_meters

    return raw_measured_pseudoranges


args: argparse.Namespace = _parse_args()

# Our ECEF position
#       (***ZIP 22102, not known at calculation time, provided to compute pseudorange to actually
#       resolve this position! ***)

hidden_actual_receiver_ecef_pos: numpy.typing.NDArray = numpy.array(
    [
         1_098_443.219,
        -4_845_862.187,
         3_985_726.416,
    ]
)


# ==========================================
# STEP 2: Input Data Provided by the Satellites
# ==========================================
# 3D ECEF positions of 4 GPS satellites all visible from 22102 at 2026-06-01 00:00:00Z (units: meters)
# These represent known coordinates extracted from the satellite ephemeris data.
satellite_positions: numpy.typing.NDArray = numpy.array(
    [
        [ 12_450_000.000, -18_340_000.000,  16_120_000.000],  # PRN 02
        [ -8_420_000.000, -22_110_000.000,  11_430_000.000],  # PRN 07
        [  5_120_000.000, -24_150_000.000,  -8_210_000.000],  # PRN 13
        [ 15_430_000.000, -11_240_000.000, -17_210_000.000]   # PRN 19
    ]
)

measured_pseudoranges: numpy.typing.NDArray = _determine_measured_pseudoranges(args,
                                                                               hidden_actual_receiver_ecef_pos,
                                                                               satellite_positions)

print()
print(f"Raw pseudoranges with unknown local clock bias included:")
print(f"-------------------------------------------------------")
print(f"\tPRN 02: {measured_pseudoranges[0]:18,.03f} m")
print(f"\tPRN 07: {measured_pseudoranges[1]:18,.03f} m")
print(f"\tPRN 13: {measured_pseudoranges[2]:18,.03f} m")
print(f"\tPRN 19: {measured_pseudoranges[3]:18,.03f} m")


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

lat, lon, alt = pymap3d.ecef2geodetic(computed_pos[0], computed_pos[1], computed_pos[2])

# ==========================================
# STEP 5: Print Final Output Metrics
# ==========================================
print()
print("             GPS PVT ENGINE OUTPUT                ")
print("==================================================")
print()
print(f"Computed Receiver Position (ECEF Meters):")
print(f"    X: {computed_pos[0]:14,.03f}")
print(f"    Y: {computed_pos[1]:14,.03f}")
print(f"    Z: {computed_pos[2]:14,.03f}")
print()
print(f"Computed Receiver Position (WGS84 lat/lon/alt):")
print(f"  Lat: {lat:9.04f} degrees")
print(f"  Lon: {lon:9.04f} degrees")
print(f"  Alt: {alt:7,.02f} m")
print()
print("--------------------------------------------------")
print()
print(f"Computed Receiver Clock Offset:")
print(f"   In Meters : {estimated_state[3]:19,.06f} m")
print(f"  In Seconds :           {computed_bias_seconds:9,.06f} s")
print()
print("==================================================")
print(f"Optimizer Success Status: {solver_output.success}")
print(f"Final Residual Cost:      {solver_output.cost:.4e}")