import argparse
import datetime
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
from trilateration_setup import satellite_pseudoranges

_logger = logging.getLogger("gps_ranging")
_logger.setLevel(logging.DEBUG)


SPEED_OF_LIGHT_M_PER_S: float = 299_792_458.0

# PRN is 1023 bits and it repeats every 0.001 s or 1 ms
PRN_RATE_BITS_PER_MS: int = 1_023

CA_CODE_CHIPPING_RATE: int = 1_023 * 1_000

# Minimum sane pseudorange = plane at 43,000 ft / 13_007 m and GPS sat is directly above it
MIN_VALID_PSEUDORANGE: dict[str, float] = {
    "meters"  : 20_200_000 - 13_007,
    "seconds" : (20_200_000 - 13_007) / SPEED_OF_LIGHT_M_PER_S,
}

_hidden_pseudoranges: dict[int, float] = {}

simulation_start_gps_time_at_receiver: datetime.datetime = datetime.datetime.fromisoformat(
    "2026-06-01T00:00:01.000000")

simulation_current_gps_time_at_receiver: datetime.datetime = datetime.datetime.fromisoformat(
    "2026-06-01T00:00:01.000000")


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
                    gold_code: numpy.typing.NDArray[numpy.int8]) -> numpy.typing.NDArray[numpy.int8]:

    global _hidden_pseudoranges

    # Calculate a noisy pseudorange in meters from our _actual_ receiver position to the satellite
    computed_pseudorange: float = compute_pseudorange_from_receiver_to_svn_meters(svn_num,
                                                                                  args.ionosphere_delay_seconds,
                                                                                  args.troposphere_delay_seconds)

    _hidden_pseudoranges[svn_num] = computed_pseudorange

    full_over_the_air_transmission_delay_seconds: float = _hidden_pseudoranges[svn_num] / SPEED_OF_LIGHT_M_PER_S
    # _logger.debug(f"OTA delay for SVN {svn_num:02d}: {full_over_the_air_transmission_delay_seconds:5.03f} s")
    transmission_delay_offset_bits: int = math.ceil((1023*1000) *
                                                    full_over_the_air_transmission_delay_seconds) % 1_023

    # _logger.debug(f"Code shift for PRN stream: {transmission_delay_offset_bits:5,} bits")

    # raise NotImplementedError("Not a thing yet")

    # Modulo so we have less iterations if it went over 1,023
    shifted_code: numpy.typing.NDArray[numpy.int8] = numpy.roll(gold_code,
                                                                -(transmission_delay_offset_bits
                                                                  % len(gold_code)))

    return shifted_code


def _pseudorange_by_received_prn_stream_bits(args: argparse.Namespace,
                                  svn_num: int,
                                  gold_code: numpy.typing.NDArray[numpy.int8]) -> int:

    sniffed_prn_stream: numpy.typing.NDArray[numpy.int8] = _svn_prn_stream(args, svn_num, gold_code)

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

    return bits_shifted_to_align


def _lnav_data_bit_sync_offset(args: argparse.Namespace, prn_num: int, sim_prn_start_offset: float) -> None:
    global simulation_current_gps_time_at_receiver

    sim_relative_time: float = (simulation_current_gps_time_at_receiver -
                                simulation_start_gps_time_at_receiver).total_seconds()

    print(f"\t\t\tSim time when receiver starts trying to find bit sync with PRN {prn_num:02} : "
          f"{sim_relative_time:.06f} s")

    sim_relative_time += sim_prn_start_offset

    print(f"\t\t\tSim time of first opportunity to test for bit sync with PRN {prn_num:02d}    : "
          f"{sim_relative_time:.06f} s")

    # Get the pseudorange computed earlier for this satellite
    satellite_pseudorange_meters: float = _hidden_pseudoranges[prn_num]

    # Find out where we are in the bitstream based on full transmission delay
    full_over_the_air_transmission_delay_seconds: float = _hidden_pseudoranges[prn_num] / SPEED_OF_LIGHT_M_PER_S

    transmission_delay_offset_chips: int = math.ceil( (1_023 * 1_000) * full_over_the_air_transmission_delay_seconds)

    _logger.debug(f"Receiver is reading data sent by PRN {prn_num:02d} {transmission_delay_offset_chips:,} chips / "
                  f"{full_over_the_air_transmission_delay_seconds:.06f} seconds ago")

    # Shift that to GPS satellite time reference
    satellite_transmission_time = simulation_current_gps_time_at_receiver - datetime.timedelta(
        seconds=full_over_the_air_transmission_delay_seconds)

    _logger.debug(f"Simulated time at GPS receiver: {simulation_current_gps_time_at_receiver.isoformat(
        timespec="microseconds")} (GPS time)")

    _logger.debug(f"Time at GPS satellite when bits were sent: {satellite_transmission_time.isoformat(
        timespec="microseconds")} (GPS time)")

    # Next bit transition will be on next 0.020s boundary (50 bits per second, starting at 0.000, last is 0.980)
    satellite_milliseconds: int  = satellite_transmission_time.microsecond // 1000
    # _logger.debug(f"Current satellite millisecond: {satellite_milliseconds}")
    milliseconds_to_next_lnav_data_bit_transition: int = 20 - (satellite_milliseconds % 20)
    next_lnav_data_bit: datetime.datetime = satellite_transmission_time + datetime.timedelta(
        milliseconds=milliseconds_to_next_lnav_data_bit_transition)

    _logger.debug(f"*Satellite* time of next LNAV data bit transition: "
                  f"{next_lnav_data_bit.isoformat(timespec="microseconds")} (GPS time)")
    delay_to_next_lnav_bit: float = (next_lnav_data_bit - satellite_transmission_time).total_seconds()

    # NO NO NO NO NO
    _logger.debug(f"*Satellite* time until next LNAV data bit transition: "
                  f"{delay_to_next_lnav_bit} s")

    simulated_number_of_lnav_data_bits_until_edge_seen: int = random.randint(0, 3)
    _logger.debug(f"Simulating that it takes {simulated_number_of_lnav_data_bits_until_edge_seen} "
                   "LNAV data bits until we see an edge")

    # Fast forward sim time to point the first LNAV bit start, even if it doesn't have an edge we can latch on
    sim_relative_time += delay_to_next_lnav_bit
    print(f"\t\t\tSim time of first LNAV message start                              : "
          f"{sim_relative_time:.06f} s")

    satellite_transmission_time += datetime.timedelta(seconds=delay_to_next_lnav_bit)

    print(f"\t\t\tSat time of first LNAV message start                              : "
          f"{satellite_transmission_time.isoformat(sep=" ", timespec="microseconds")} GPS")

    time_offset_from_first_edge_check_to_first_data_bit_transition_seconds: float = \
        (0.02 * simulated_number_of_lnav_data_bits_until_edge_seen)

    # Number of signal samples on PRN code boundaries before we saw a data bit edge
    # _logger.debug(f"Millis to next bit: {milliseconds_to_next_lnav_data_bit_transition}")
    number_checks_for_data_bit_edge: int = milliseconds_to_next_lnav_data_bit_transition + (
        20 * simulated_number_of_lnav_data_bits_until_edge_seen)

    _logger.debug(f"Receiver sniffed at C/A period boundaries {number_checks_for_data_bit_edge} times "
                   "until the first data bit start seen")

    simulation_current_gps_time_at_receiver += datetime.timedelta(
        seconds=time_offset_from_first_edge_check_to_first_data_bit_transition_seconds)

    sim_relative_time += time_offset_from_first_edge_check_to_first_data_bit_transition_seconds

    print(f"\t\t\tSim time when bit sync was established with PRN {prn_num:02d}                : "
          f"{sim_relative_time:.06f} s")

    simulation_current_gps_time_at_receiver = simulation_start_gps_time_at_receiver + datetime.timedelta(
        seconds=sim_relative_time)

    satellite_transmission_time += datetime.timedelta(
        seconds=time_offset_from_first_edge_check_to_first_data_bit_transition_seconds)

    print(f"\t\t\tSat time when bit sync was established with PRN {prn_num:02d}                : "
          f"{satellite_transmission_time.isoformat(sep=" ", timespec="microseconds")} s")


def _get_sim_time_offsets_lnav_bit_starts() -> list[float]:
    curr_sim_time_bit_start: float = (simulation_current_gps_time_at_receiver -
                                      simulation_start_gps_time_at_receiver).total_seconds()

    while curr_sim_time_bit_start > 0.020:
        curr_sim_time_bit_start -= 0.020

    bit_start_offsets: list[float] = [
        curr_sim_time_bit_start,
    ]

    while (curr_sim_time_bit_start := curr_sim_time_bit_start + 0.020) < 1.0:
        bit_start_offsets.append(curr_sim_time_bit_start)

    return bit_start_offsets


def _read_time_of_week(args: argparse.Namespace, prn_num: int) -> datetime.datetime:
    global simulation_current_gps_time_at_receiver

    sim_relative_time: float = (simulation_current_gps_time_at_receiver -
                                simulation_start_gps_time_at_receiver).total_seconds()

    print(f"\t\t\tSim time at start of Time of Week (TOW) search : {sim_relative_time:.06f} s")

    print(f"\t\t\tReceiver absolute time: {simulation_current_gps_time_at_receiver.isoformat(
        sep=" ", timespec="microseconds")} (GPS)")

    satellite_pseudorange_meters: float = _hidden_pseudoranges[prn_num]

    # Full transmission delay
    full_over_the_air_transmission_delay_seconds: float = _hidden_pseudoranges[prn_num] / SPEED_OF_LIGHT_M_PER_S

    # compute satellite time when bits were being sent
    sat_gps_time: datetime.datetime = simulation_current_gps_time_at_receiver - datetime.timedelta(
        seconds=full_over_the_air_transmission_delay_seconds)

    print(f"\t\t\tSatellite time: {sat_gps_time.isoformat(sep=" ", timespec="microseconds")} (GPS)")



def _main() -> None:
    global simulation_current_gps_time_at_receiver

    args: argparse.Namespace = _parse_args()

    print()

    # Populate the gold codes for the satellites we're simulating a lock to
    print("Generating GPS Gold codes for satellites we are going to use in our position fix")
    gold_codes: gps_gold_codes.GPSGoldCodes = gps_gold_codes.GPSGoldCodes()
    fix_svn_prns: dict[int, numpy.typing.NDArray[numpy.int8]] = {}
    for svn_num in (2, 7, 13, 19):
        fix_svn_prns[svn_num] = gold_codes.gold_code(svn_num)
    print("\tDone!")

    print()
    print("Calculating pseudoranges for all satellites in the fix")

    for svn_num in (2, 7, 13, 19):

        print()
        print(f"\tPseudoranging to SVN {svn_num:02}")

        print(f"\n\t\t*** Simulation *GPS* time at receiver: "
              f"{simulation_current_gps_time_at_receiver.isoformat(sep=" ", timespec="microseconds")} ***")

        print()
        print(f"\t\tEstablishing local clock offset to PRN {svn_num:02} L1 C/A signal (PRN code alignment)")
        pseudorange_in_bits_per_svn: dict[int, int] = {}

        pseudorange_in_bits: int = _pseudorange_by_received_prn_stream_bits(args, svn_num, fix_svn_prns[svn_num])
        pseudorange_in_bits_per_svn[svn_num] = pseudorange_in_bits

        time_for_prn_sync_seconds: float = 0.001
        simulation_current_gps_time_at_receiver += datetime.timedelta(seconds=time_for_prn_sync_seconds)
        print(f"\t\t\tSim time when PRN {svn_num:02d} sync was established       : 0.001000 s")
        sim_prn_start_offset: float = pseudorange_in_bits_per_svn[svn_num] / CA_CODE_CHIPPING_RATE
        print(f"\t\t\tSim time sub-millisecond offset when PRN starts : "
              f"{sim_prn_start_offset:.06f} s")

        print(f"\n\t\t*** Simulation *GPS* time at receiver: "
              f"{simulation_current_gps_time_at_receiver.isoformat(sep=" ", timespec="microseconds")} ***")

        # Now we have locked sync on the most fundamental signal, now establish sync to start of each
        #   bit of the LNAV message, so we can start to read LNAV messages.
        #
        # We needed C/A code alignment as a prereq as it tells us time window when LNAV data bit flips *may*
        #   happen (every 20 repetitions of full C/A PRN, e.g. 0.020 s)
        print()
        print(f"\t\tEstablishing time alignment with the exact start of each data bit in PRN {svn_num:02d} LNAV messages "
               "(\"bit sync\")")
        _lnav_data_bit_sync_offset(args, svn_num, sim_prn_start_offset)

        sim_time_offsets_lnav_bit_starts: list[float] = _get_sim_time_offsets_lnav_bit_starts()

        print(f"\t\t\tSim time sub-second offsets of LNAV bit starts: {sim_time_offsets_lnav_bit_starts[0]:8.06f} s "
              "+ integer multiples of 0.020 s (20 ms)")

        print(f"\n\t\t*** Simulation *GPS* time at receiver: "
              f"{simulation_current_gps_time_at_receiver.isoformat(sep=" ", timespec="microseconds")} ***")

        print()
        print("\t\tReading Time Of Week (TOW) data")
        svn_time_of_week = datetime.datetime = _read_time_of_week(args, svn_num)

        print(f"\n\t\t*** Simulation *GPS* time at receiver: "
              f"{simulation_current_gps_time_at_receiver.isoformat(sep=" ", timespec="microseconds")} ***")

        print()
        print("\t\tCalculating pseudorange now that we have TOW + time since TOW + fractional offset")
        print("\t\t\tDone!")

        break

    print()


if __name__ == "__main__":
    logging.basicConfig()
    _main()
