import argparse
import random
import time

import numpy
import numpy.typing
import pymap3d
import scipy.optimize

import trilateration_setup


_number_least_squares_iterations: int = 0


def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Trilateration demo for GPS")
    parser.add_argument("-b", "--receiver-clock-bias", type=float,
                        # Random distance bias of -300,000 .. +300,000 m
                        default=(random.random() * 0.002) - 0.001,
                        help="Local clock offset from GPS time in seconds (-0.001 .. +0.001) (default: random)")
    parser.add_argument("-i", "--ionosphere-delay-seconds", type=float,
                        help="Fixed ionospheric propagation delay (typical: 0.0000000007 - 0.000001600 s / "
                             "0.7 - 1,600 ns)"
                             "(default: random per satellite)")
    parser.add_argument("-t", "--troposphere-delay-seconds", type=float,
                        help="Fixed tropospheric propagation delay (typical: 0.000000008 - 0.000000100 s / "
                             "8 - 100 ns)"
                             "(default: random per satellite)")
    return parser.parse_args()


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

    print()
    print("Simulated fix using GPS satellites visible from Fairfax County, VA, USA at 2026-06-01 00:00 UTC")
    print()

    satellite_positions: numpy.typing.NDArray = trilateration_setup.satellite_ecef_positions()

    measured_pseudoranges: numpy.typing.NDArray = trilateration_setup.satellite_pseudoranges(
        args.receiver_clock_bias, satellite_positions, args.ionosphere_delay_seconds,
        args.troposphere_delay_seconds)

    print()
    print(f"Receiver-measured pseudoranges")
    print()
    print(f"\tPRN  Pseudorange (m)")
    print(f"\t---  ---------------")
    for i in range(len(measured_pseudoranges)):
        print(f"\t {trilateration_setup.satellite_prn_numbers[i]}  {measured_pseudoranges[i]:15,.03f}")
    print()

    print("\tThese pseudoranges are initialized to the *actual* distance between the receiver and\n"
          "\tposition of each bird (provided by the highly accurate GPS ephemeris data the receiver\n"
          "\twould have downloaded before being able to attempt trilateration) obfuscated by clock bias\n"
          "\tand atmospheric propagation delays.\n"
          "\n"
          "\tNote that this is realistic situation -- the pseudoranges from actual receivers show the\n"
          "\ttransmission delay for each bird, this is actually a legit simulation, not a totally\n"
          "\tfaked-out demo.")

    print()
    print()
    print(f"Four equations with four unknowns being passed to the equation solver")

    for sat_num in range(len(satellite_positions)):
        print(f"\n\t{measured_pseudoranges[sat_num]:14,.03f} = sqrt( "
              f"({satellite_positions[sat_num][0]:15,.03f} - receiver_x)**2 + \n"
              f"\t                       ({satellite_positions[sat_num][1]:15,.03f} - receiver_y)**2 + \n"
              f"\t                       ({satellite_positions[sat_num][2]:15,.03f} - receiver_z)**2\n"
              f"\t                     ) + receiver_clock_bias_distance")

    # The cold start: we assume the receiver is at the exact center of the Earth
    # with a perfectly synchronized clock. No hidden data is referenced.
    cold_start_guess: numpy.typing.NDArray = numpy.array([0.0, 0.0, 0.0, 0.0])

    print("\n\nRunning equation solver (iterative least squares, Levenberg-Marquardt algorithm)")
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
    computed_bias_seconds: float = estimated_state[3] / trilateration_setup.SPEED_OF_LIGHT_M_PER_S

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
