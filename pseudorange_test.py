import argparse
import random
import time

import numpy
import numpy.typing
import pymap3d
import scipy.optimize


# ==========================================
# STEP 1: Speed of Light Constant
# ==========================================
# Crucial for converting the 4th dimension (meters) back into time (seconds).
_SPEED_OF_LIGHT: float = 299_792_458.0

_number_least_squares_iterations: int = 0


def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Trilateration demo for GPS")
    parser.add_argument("-b", "--receiver-clock-bias", type=float,
                        # Random distance bias of -300,000 .. +300,000 m
                        default=(random.random() * 0.002) - 0.001,
                        help="Local clock offset from GPS time in seconds (-0.001 .. +0.001) (default: random)")
    return parser.parse_args()


def _determine_measured_pseudoranges(args: argparse.Namespace,
                                     satellite_positions: numpy.typing.NDArray) -> numpy.typing.NDArray:

    clock_bias_seconds: float = args.receiver_clock_bias
    clock_bias_meters: float = clock_bias_seconds * _SPEED_OF_LIGHT

    if not -0.001 <= clock_bias_seconds <= 0.001:
        raise ValueError(f"Clock bias must be between -0.001 and +0.001 seconds to let least squares find a solution")

    print()
    print("Simulated fix using GPS satellites visible from Fairfax County, VA, USA at 2026-06-01 00:00 UTC")
    print()

    # This is only used to compute initial pseudorange, need to calculate actual signal propagation delay
    #       It is obfuscated by the clock bias so least squares needs to actually solve the equation
    hidden_actual_receiver_ecef_pos: numpy.typing.NDArray = numpy.array(
        [
             1_088_598.0,
            -4_853_018.0,
             3_979_796.0
        ]
    )

    # 3. Compute vector differences (subtract cold start receiver position from each satellite)
    delta_vectors: numpy.typing.NDArray = satellite_positions - hidden_actual_receiver_ecef_pos

    # 4. Compute true geometric range (L2 Euclidean norm along the row axis)
    geometric_ranges: numpy.typing.NDArray = numpy.linalg.norm(delta_vectors, axis=1)

    # 5. Add receiver clock bias in meters to produce obfuscated pseudoranges
    raw_measured_pseudoranges: numpy.typing.NDArray = geometric_ranges + clock_bias_meters

    print()
    print(f"Receiver-measured pseudoranges")
    print()
    print(f"\tPRN  Pseudorange (m)  Actual range (m)   Local Clock Bias (m)")
    print(f"\t---  ---------------  ----------------   --------------------")
    print(f"\t 02  {raw_measured_pseudoranges[0]:15,.03f}  "
          f"({geometric_ranges[0]:15,.03f} + {clock_bias_meters:19,.03f})")
    print(f"\t 07  {raw_measured_pseudoranges[1]:15,.03f}  "
          f"({geometric_ranges[1]:15,.03f} + {clock_bias_meters:19,.03f})")
    print(f"\t 13  {raw_measured_pseudoranges[2]:15,.03f}  "
          f"({geometric_ranges[2]:15,.03f} + {clock_bias_meters:19,.03f})")
    print(f"\t 19  {raw_measured_pseudoranges[3]:15,.03f}  "
          f"({geometric_ranges[3]:15,.03f} + {clock_bias_meters:19,.03f})")
    print()

    print("\tThese pseudoranges are initialized to the *actual* distance between the receiver and\n"
          "\tposition of each bird (provided by the highly accurate GPS ephemeris data the receiver\n"
          "\twould have downloaded before being able to attempt trilateration) plus clock bias.\n"
          "\n"
          "\tNote that this is realistic situation -- the pseudoranges from actual receivers show the\n"
          "\ttransmission delay for each bird, this is actually a legit simulation, not a totally\n"
          "\tfaked-out demo.")

    return raw_measured_pseudoranges



def gps_residuals(
        state_guess: numpy.typing.NDArray,
        sat_pos: numpy.typing.NDArray,
        pseudoranges: numpy.typing.NDArray) -> numpy.typing.NDArray:

    # print(f"\t\tEntering least squares iteration with state guess {state_guess}")

    global _number_least_squares_iterations

    _number_least_squares_iterations += 1

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


def _main() -> None:
    args: argparse.Namespace = _parse_args()

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
                                                                                   satellite_positions)

    print()
    print()
    print(f"Four equations with four unknowns we are going to ask least squares to find a solution to ")

    for sat_num in range(len(satellite_positions)):
        print(f"\n\t{measured_pseudoranges[sat_num]:14,.03f} = sqrt( "
              f"({satellite_positions[sat_num][0]:15,.03f} - receiver_x)**2 + \n"
              f"\t                       ({satellite_positions[sat_num][1]:15,.03f} - receiver_y)**2 + \n"
              f"\t                       ({satellite_positions[sat_num][2]:15,.03f} - receiver_z)**2\n"
              f"\t                     ) + receiver_clock_bias_distance")

    # The cold start: we assume the receiver is at the exact center of the Earth
    # with a perfectly synchronized clock. No hidden data is referenced.
    cold_start_guess: numpy.typing.NDArray = numpy.array([0.0, 0.0, 0.0, 0.0])

    print("\n\nRunning equation solver (least squares, Levenberg-Marquardt algorithm)")
    print("\n\tAlgorithm inputs provided:")
    print("\t\t- Cold start position guess (ECEF = (0.0, 0.0, 0.0))")
    print("\t\t- Cold start receiver clock bias estimate (0.0 meters)")
    print("\t\t- Exact ECEF positions of four satellites used in fix from GPS ephemeris")
    print("\t\t- Four measured pseudoranges to the four satellites in meters")


    # Solve the non-linear system of equations
    start_time: float = time.perf_counter()
    solver_output = scipy.optimize.least_squares(
        gps_residuals,
        cold_start_guess,
        args=(satellite_positions, measured_pseudoranges),
        method="lm" # Levenberg-Marquardt algorithm is ideal for tracking/least-squares
    )
    end_time: float = time.perf_counter()
    print()
    print("\tLeast squares iterations")
    print(f"\t\t    Number of iterations : {_number_least_squares_iterations:,}")
    print(f"\t\t              Clock time : {end_time - start_time:5.03f} s")

    print()
    print("\tLeast squares output")
    print(f"\t\tOptimizer Success Status : {solver_output.success}")
    print(f"\t\t     Final Residual Cost : {solver_output.cost:.4e}")

    estimated_state = solver_output.x
    computed_pos: numpy.typing.NDArray = estimated_state[0:3]
    computed_bias_seconds: float = estimated_state[3] / _SPEED_OF_LIGHT

    lat, lon, alt = pymap3d.ecef2geodetic(computed_pos[0], computed_pos[1], computed_pos[2])

    print()
    print()
    print(f"Estimated Receiver Position & Clock Bias")
    print(f"\n\tPosition")
    print(f"\n\t\tECEF")
    print(f"\t\t\t     X :  {computed_pos[0]:14,.03f} m")
    print(f"\t\t\t     Y :  {computed_pos[1]:14,.03f} m")
    print(f"\t\t\t     Z :  {computed_pos[2]:14,.03f} m")
    print()
    print(f"\t\tWGS84")
    print(f"\t\t\t   Lat : {lat:9.04f} degrees")
    print(f"\t\t\t   Lon : {lon:9.04f} degrees")
    print(f"\t\t\t   Alt :  {alt:5,.01f} m")
    print()
    print(f"\tTime")
    print(f"\n\t\tReceiver clock offset/bias: {computed_bias_seconds:.06f} s")
    print()


if __name__ == "__main__":
    _main()
