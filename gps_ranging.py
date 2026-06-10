import argparse
import json
import logging
import math
import random
import time
import typing

import numpy
import numpy.linalg
import numpy.typing

import gps_gold_codes
import gps_ephemeris
import gps_receiver

_logger = logging.getLogger("gps_ranging")
_logger.setLevel(logging.DEBUG)


SPEED_OF_LIGHT_M_PER_S: float = 299_792_458.0

# PRN is 1023 bits and it repeats every 0.001 s or 1 ms
PRN_RATE_BITS_PER_MS: int = 1_023

# Minimum sane pseudorange = plane at 43,000 ft / 13_007 m and GPS sat is directly above it
MIN_VALID_PSEUDORANGE: dict[str, float] = {
    "meters"  : 20_200_000 - 13_007,
    "seconds" : (20_200_000 - 13_007) / SPEED_OF_LIGHT_M_PER_S,
}


def compute_pseudorange_from_receiver_to_svn_meters(svn_num: int,
                                                    ionosphere_delay_seconds: float | None,
                                                    tropospheric_delay_seconds: float | None) -> float:

    # Normalization gets _length_ of vector
    actual_transmission_delay_meters: float = float(
        numpy.linalg.norm(
            gps_ephemeris.gps_satellite_positions_by_svn[svn_num] - gps_receiver.actual_receiver_ecef_pos
        )
    )

    # Add in atmospheric delays
    atmospheric_delays: float = 0.0
    if ionosphere_delay_seconds is None:
        ionosphere_delay_seconds = float(random.randint(1, 1601)) * 0.000_000_001
        # _logger.debug(f" Random ionospheric delay : {ionosphere_delay_seconds:.09f} s")

    if tropospheric_delay_seconds is None:
        tropospheric_delay_seconds = float(random.randint(8, 101)) * 0.000_000_001
        # _logger.debug(f"Random tropospheric delay : {tropospheric_delay_seconds:.09f} s")

    computed_pseudorange_meters: float = actual_transmission_delay_meters + (
            (ionosphere_delay_seconds + tropospheric_delay_seconds) * SPEED_OF_LIGHT_M_PER_S
    )

    _logger.debug(f"SVN {svn_num:02d} pseudorange: {computed_pseudorange_meters:,.02f} m = "
                  f"{actual_transmission_delay_meters:,.02f} (actual) + " 
                  f"{ionosphere_delay_seconds * SPEED_OF_LIGHT_M_PER_S:6.02f} (ionospheric) + "
                  f"{tropospheric_delay_seconds * SPEED_OF_LIGHT_M_PER_S:5.02f} (tropospheric)"
    )

    return computed_pseudorange_meters


def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Trilateration demo for GPS")
    parser.add_argument("-b", "--prn-period-offset-milliseconds", type=int,
                        # Random -1 .. +1 ms
                        default=random.randint(0, 999),
                        help="PRN period millisecond offset (0 .. 999 ms) (default: random)")
    parser.add_argument("-i", "--ionosphere-delay-seconds", type=float,
                        help="Fixed ionospheric propagation delay (typical: 0.0000000007 - 0.000001600 s / "
                             "0.7 - 1,600 ns)"
                             "(default: random per satellite)")
    parser.add_argument("-t", "--troposphere-delay-seconds", type=float,
                        help="Fixed tropospheric propagation delay (typical: 0.000000008 - 0.000000100 s / "
                             "8 - 100 ns)"
                             "(default: random per satellite)")
    return parser.parse_args()


def _svn_prn_stream(args: argparse.Namespace,
                    svn_num: int,
                    fix_svn_prns: dict[int, numpy.typing.NDArray[numpy.int8]]) -> numpy.typing.NDArray[numpy.int8]:

    # Calculate a noisy pseudorange in meters from our _actual_ receiver position to the satellite
    computed_pseudorange: float = compute_pseudorange_from_receiver_to_svn_meters(svn_num,
                                                                                  args.ionosphere_delay_seconds,
                                                                                  args.troposphere_delay_seconds)

    full_over_the_air_transmission_delay_seconds: float = computed_pseudorange / SPEED_OF_LIGHT_M_PER_S
    # _logger.debug(f"OTA delay for SVN {svn_num:02d}: {full_over_the_air_transmission_delay_seconds:5.03f} s")
    transmission_delay_offset_bits: int = int(full_over_the_air_transmission_delay_seconds * PRN_RATE_BITS_PER_MS)

    # Shift stream by offset into millisecond that receiver starts listening
    period_offset_bits: int = int((float(args.prn_period_offset_milliseconds) / 1_000.0) *
                                  PRN_RATE_BITS_PER_MS)

    total_code_shift_bits: int = transmission_delay_offset_bits + period_offset_bits
    _logger.debug(f"Total code shift for PRN stream: {total_code_shift_bits:5,} "
                  f"({transmission_delay_offset_bits:3d} OTA + {period_offset_bits:5,} period offset)")

    # raise NotImplementedError("Not a thing yet")

    # Modulo so we have less iterations if it went over 1,023
    shifted_code: numpy.typing.NDArray[numpy.int8] = numpy.roll(fix_svn_prns[svn_num],
                                                                -(total_code_shift_bits % len(fix_svn_prns[svn_num])))

    return shifted_code


def _pseudorange_by_received_prn_stream(args: argparse.Namespace,
                                  svn_num: int,
                                  fix_svn_prns: dict[int, numpy.typing.NDArray[numpy.int8]],
                                  gold_code: numpy.typing.NDArray[numpy.int8]) -> float:
    sniffed_prn_stream: numpy.typing.NDArray[numpy.int8] = _svn_prn_stream(args, svn_num, fix_svn_prns)

    # Find code shift to line up
    bits_shifted_to_align: int = 0
    curr_test_bits: numpy.typing.NDArray[numpy.int8] = gold_code
    while bits_shifted_to_align < 1_024:
        if numpy.array_equal(curr_test_bits, sniffed_prn_stream):
            break
        curr_test_bits = numpy.roll(curr_test_bits, -1)
        bits_shifted_to_align += 1
    else:
        raise RuntimeError("Tried 1,023 shifts and did not line up!")

    _logger.debug(f"Got alignment on sniffed PRN stream and gold code at shift #{bits_shifted_to_align:5,}")


def _main() -> None:
    args: argparse.Namespace = _parse_args()

    print()

    # Populate the gold codes for the satellites we're simulating a lock to
    print("Generating GPS Gold codes for satellites in our position fix")
    gold_codes: gps_gold_codes.GPSGoldCodes = gps_gold_codes.GPSGoldCodes()
    fix_svn_prns: dict[int, numpy.typing.NDArray[numpy.int8]] = {}
    for svn_num in (2, 7, 13, 19):
        fix_svn_prns[svn_num] = gold_codes.gold_code(svn_num)
    print("\tDone!")

    print()
    print("\"Listening\" to PRN stream from our four satellites for one full PRN repetitions (1 ms)")
    for svn_num in (2, 7, 13, 19):
        pseudorange_in_meters: float = _pseudorange_by_received_prn_stream(args, svn_num, fix_svn_prns,
                                                                           fix_svn_prns[svn_num])



    print("\tDone!")

    print()


if __name__ == "__main__":
    logging.basicConfig()
    _main()
