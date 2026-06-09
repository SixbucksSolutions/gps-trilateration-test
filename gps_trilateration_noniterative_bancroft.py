import argparse
import json
import random
import time
import typing

import numpy
import numpy.typing
import pymap3d


import trilateration_setup


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


def _bancroft_trilateration(sat_positions: numpy.typing.NDArray,
                            pseudoranges: numpy.typing.NDArray) -> tuple[numpy.typing.NDArray, float]:
    """
    Performs non-iterative GPS trilateration cleanly using Bancroft's Algorithm.

    Parameters:
    sat_positions (numpy.ndarray): Nx3 array of satellite coordinates (X, Y, Z).
    pseudoranges (numpy.ndarray): Nx1 (or 1D array of length N) of pseudoranges.

    Returns:
    numpy.ndarray: 3-element array of receiver coordinates (X, Y, Z).
    float: Receiver clock bias (in meters).
    """
    # Ensure proper array shapes
    A = numpy.atleast_2d(sat_positions)
    r = numpy.atleast_2d(pseudoranges).reshape(-1, 1)
    n = A.shape[0]

    if n < 4:
        raise ValueError("At least 4 satellites are required for GPS trilateration.")

    # Step 1: Compute Lorentz constant 'alpha' for each satellite row
    # alpha_i = 0.5 * (x_i^2 + y_i^2 + z_i^2 - r_i^2)
    alpha = 0.5 * (numpy.sum(A ** 2, axis=1, keepdims=True) - r ** 2)

    # Step 2: Construct Matrix B (CRITICAL: Pseudoranges must be NEGATIVE)
    B = numpy.hstack((A, -r))
    B_pinv = numpy.linalg.pinv(B)

    # Step 3: Compute standard components g1 and g2
    ones = numpy.ones((n, 1))
    g1 = B_pinv @ alpha
    g2 = B_pinv @ ones

    # Step 4: Define the Minkowski Lorentz Matrix M = diag(1, 1, 1, -1)
    M = numpy.diag([1.0, 1.0, 1.0, -1.0])

    def lorentz_dot(u, v):
        # Explicit matrix multiply and extraction to handle NumPy 1.25+ scalar rules
        return float((u.T @ M @ v).item())

    # Step 5: Compute correct quadratic formula coefficients
    a = lorentz_dot(g2, g2)
    b = 2 * (lorentz_dot(g1, g2) - 1.0)
    c = lorentz_dot(g1, g1)

    # Solve the quadratic formula for lambda
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No physical solution found (negative discriminant). Check inputs.")

    sqrt_disc = numpy.sqrt(discriminant)
    lambda1 = (-b + sqrt_disc) / (2 * a)
    lambda2 = (-b - sqrt_disc) / (2 * a)

    # Step 6: Compute the two possible physical 4-vector solutions
    y1 = g1 + lambda1 * g2
    y2 = g1 + lambda2 * g2

    # Extract positions
    pos1 = y1[:3, 0]
    pos2 = y2[:3, 0]

    # Earth radius approximation (~6,371,000 meters)
    r_earth = 6_371_000.0

    solutions: list[dict[str, float | numpy.typing.NDArray]] = [
        {
            "distance_from_surface" : abs(float(numpy.linalg.norm(pos1)) - r_earth),
            "ecef_pos"              : pos1,
            "clock_bias_meters"     : -y1[3, 0],
        },

        {
            "distance_from_surface" : abs(float(numpy.linalg.norm(pos2)) - r_earth),
            "ecef_pos"              : pos2,
            "clock_bias_meters"     : -y2[3, 0],
        },
    ]

    # print(json.dumps(solutions, indent=4, sort_keys=True, default=str))

    if solutions[0]["distance_from_surface"] < solutions[1]["distance_from_surface"]:
        selected_solution: int = 0
    else:
        selected_solution: int = 1

    return typing.cast(
        tuple[numpy.typing.NDArray, float],
        (
            solutions[selected_solution]["ecef_pos"],
            solutions[selected_solution]["clock_bias_meters"]
        )
    )


def _main() -> None:
    args: argparse.Namespace = _parse_args()

    print()
    print("Simulated fix using GPS satellites visible from Fairfax County, Virginia, USA at 2026-06-01 00:00 UTC")
    print()

    satellite_positions: numpy.typing.NDArray = trilateration_setup.satellite_ecef_positions()

    measured_pseudoranges: numpy.typing.NDArray = trilateration_setup.satellite_pseudoranges(
        args.receiver_clock_bias, satellite_positions, args.ionosphere_delay_seconds,
        args.troposphere_delay_seconds)

    print()
    print(f"Receiver-measured pseudoranges")
    print()

    if args.ionosphere_delay_seconds is not None:
        print(f"\tUsed fixed ionospheric propagation delay of {args.ionosphere_delay_seconds:.09f}")
    else:
        print("\tUsed random per-satellite ionospheric propagation delay values")

    if args.troposphere_delay_seconds is not None:
        print(f"\tUsed fixed tropospheric propagation delay of {args.troposphere_delay_seconds:.09f}")
    else:
        print("\tUsed random per-satellite tropospheric propagation delay values")

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

    print("\n\n")
    print("Running Bancroft non-iterative/closed trilateration algorithm")
    print("\n\tAlgorithm inputs provided:")
    print("\t\t- Exact ECEF positions of four satellites used in fix from GPS ephemeris")
    print("\t\t- Four measured pseudoranges to the four satellites in meters")

    start_time: float = time.perf_counter()
    computed_pos, computed_bias_meters = _bancroft_trilateration(satellite_positions, measured_pseudoranges)
    computed_bias_seconds: float = computed_bias_meters / trilateration_setup.SPEED_OF_LIGHT_M_PER_S
    end_time: float = time.perf_counter()
    print(f"\n\tClock time: {end_time - start_time:.06f} s")

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

    fix_error_position_meters: float = trilateration_setup.fix_error_position_meters(computed_pos)
    # print(f"{args.receiver_clock_bias} / {computed_bias_seconds}")
    fix_error_receiver_clock_offset_seconds: float = abs(computed_bias_seconds - args.receiver_clock_bias)

    print()
    print()
    print(f"Error in First Fix")
    print(f"\n\t          3D Position : {fix_error_position_meters:5.1f} m")
    print(  f"\tReceiver clock offset : {fix_error_receiver_clock_offset_seconds:13.09f} s")

    print()

if __name__ == "__main__":
    _main()
