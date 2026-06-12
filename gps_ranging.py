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
import sim_time

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

MICROSECONDS_PER_SECOND: int = 1_000_000

_hidden_pseudoranges: dict[int, float] = {}


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



def _determine_sat_hidden_pseudorange(args: argparse.Namespace, prn_num: int) -> None:
    global _hidden_pseudoranges

    # Calculate a noisy pseudorange in meters from our _actual_ receiver position to the satellite
    computed_pseudorange: float = compute_pseudorange_from_receiver_to_svn_meters(prn_num,
                                                                                  args.ionosphere_delay_seconds,
                                                                                  args.troposphere_delay_seconds)

    _hidden_pseudoranges[prn_num] = computed_pseudorange
    _logger.info(f"Set hidden pseudorange for PRN {prn_num:02d} to {_hidden_pseudoranges[prn_num]:,.02f} m")


def _svn_prn_stream(args: argparse.Namespace,
                    svn_num: int,
                    gold_code: numpy.typing.NDArray[numpy.int8]) -> numpy.typing.NDArray[numpy.int8]:


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


def _establish_bit_sync(args: argparse.Namespace, sim_engine: sim_time.SimTime,
                        prn_num: int, prn_sync_offset: float) -> None:

    _logger.debug(f"Starting to establish bit sync")

    satellite_pseudorange_meters: float = _hidden_pseudoranges[prn_num]

    # Find out where we are in the bitstream based on full transmission delay
    full_over_the_air_transmission_delay_seconds: float = _hidden_pseudoranges[prn_num] / SPEED_OF_LIGHT_M_PER_S

    transmission_delay_offset_chips: int = math.ceil( (1_023 * 1_000) * full_over_the_air_transmission_delay_seconds)

    # Compute sat delta to next 20ms interval (time instants when LNAV bits start)
    sv_clock: sim_time.SimTime.UserClock = sim_engine.user_clock("prn02_absolute_gps")
    sv_abs_time: datetime.datetime = sv_clock.time_current()
    current_second_us: int = (sv_abs_time.minute * 60 * 1_000_000) + (sv_abs_time.second * 1_000_000) + \
                              sv_abs_time.microsecond

    # Calculate microseconds needed to reach the next 20ms interval
    interval_us: int = 20_000
    remainder_us: int = current_second_us % interval_us
    delay_to_next_bit_transition_edge: datetime.timedelta = datetime.timedelta(
        microseconds=interval_us - remainder_us)

    number_data_bits_until_edge_seen: int
    if (number_data_bits_until_edge_seen := random.randint(0, 3)) > 0:
        delay_to_next_bit_transition_edge += datetime.timedelta(seconds=0.020 * number_data_bits_until_edge_seen)

    # Advance sim time to first data bit transition edge we see in this sim run
    _logger.info(
        f"Bit sync: advancing clock {sim_time.SimTime.timedelta_isoformat(delay_to_next_bit_transition_edge)} s "
         "as it's when bit sync is established (receiver sees its first (0 <-> 1) transition edge at a data bit boundary"
    )
    sim_engine.advance_sim_time(delay_to_next_bit_transition_edge)

    raise NotImplementedError("Stop")

    simulated_number_of_lnav_data_bits_until_edge_seen: int = random.randint(0, 3)

    if simulated_number_of_lnav_data_bits_until_edge_seen > 0:
        _logger.debug(f"Simulating that it takes {simulated_number_of_lnav_data_bits_until_edge_seen} "
                      "LNAV data bits until we see a (0 <-> 1) transition")
        clock_step: datetime.timedelta = datetime.timedelta(
            milliseconds=simulated_number_of_lnav_data_bits_until_edge_seen * 20)
        fractional_second_clock_step = float(clock_step.microseconds) / MICROSECONDS_PER_SECOND
        _logger.info(f"Advancing all simulation times by {fractional_second_clock_step:.06f} s to "
                     f"{(sat_abs_time + clock_step).isoformat(sep=" ", timespec="microseconds")} (GPS) "
                      "where the first LNAV message bit transition edge occurs")

        sat_abs_time += clock_step
        sim_relative_time += fractional_second_clock_step
        simulation_current_gps_time_at_receiver += clock_step


        _logger.debug(f"Sim absolute time: {simulation_current_gps_time_at_receiver.isoformat(sep=" ", 
                                                                                            timespec="microseconds")}")
        _logger.debug(f"Sim relative time: {sim_relative_time:.06f} s")
        _logger.debug(f"Sat absolute time: {sat_abs_time.isoformat(sep=" ", timespec="microseconds")}")

    else:
        _logger.debug(f"PRNG gave us a bit transition the first time we checked, no need to keep looking")


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
    _logger.info("Starting search for time of week")

    sim_relative_time: float = (simulation_current_gps_time_at_receiver -
                                simulation_start_gps_time_at_receiver).total_seconds()

    _logger.debug(f"Sim time at entry               : {sim_relative_time:.06f} s")

    _logger.debug(f"Receiver absolute time at entry : {simulation_current_gps_time_at_receiver.isoformat(
        sep=" ", timespec="microseconds")} (GPS)")

    satellite_pseudorange_meters: float = _hidden_pseudoranges[prn_num]

    # Full transmission delay
    full_over_the_air_transmission_delay_seconds: float = _hidden_pseudoranges[prn_num] / SPEED_OF_LIGHT_M_PER_S

    # compute satellite time when bits were being sent
    sat_gps_time: datetime.datetime = simulation_current_gps_time_at_receiver - datetime.timedelta(
        seconds=full_over_the_air_transmission_delay_seconds)

    _logger.debug(f"Satellite time at entry         : {sat_gps_time.isoformat(sep=" ", timespec="microseconds")} (GPS)")

    print("\t\t\tThe LNAV preamble is sent on a six second interval"
          "\n\t\t\tThe edge of the first bit of LNAV preamble is sent *exactly* at the GPS second boundary from sat"
          "\n\t\t\t\tframe of reference")

    # That means the next preamble starts at the next six-second interval

    # Calculate microseconds needed to reach the next six-second interval
    current_second_us: int = (sat_gps_time.minute * 60 * 1_000_000) + (sat_gps_time.second * 1_000_000) + \
                                    sat_gps_time.microsecond
    interval_us: int = 6_000_000
    remainder_us: int = current_second_us % interval_us
    difference_us: int = interval_us - remainder_us

    # Advance all sim times in lockstep
    clock_step: datetime.timedelta = datetime.timedelta(microseconds=difference_us)
    fractional_second_clock_step: float = float(clock_step.microseconds) / MICROSECONDS_PER_SECOND
    _logger.info(f"Advancing all simulation times by {clock_step.seconds + fractional_second_clock_step:.06f} s "
                   f"to get to next LNAV preamble bit boundary time of {(sat_gps_time + clock_step).isoformat(
                       sep=" ", timespec="microseconds")} (GPS)")

    sat_gps_time += clock_step
    sim_relative_time += clock_step.seconds + fractional_second_clock_step
    simulation_current_gps_time_at_receiver += clock_step

    _logger.debug(f"Sim absolute time: {simulation_current_gps_time_at_receiver.isoformat(sep=" ",
                                                                                          timespec="microseconds")}")
    _logger.debug(f"Sim relative time: {sim_relative_time:.06f} s")
    _logger.debug(f"Sat absolute time: {sat_gps_time.isoformat(sep=" ", timespec="microseconds")}")

    print( "\t\t\tNext preamble found at: "
          f"\n\t\t\t\tSim time "
          f"{simulation_current_gps_time_at_receiver.isoformat(sep=" ", timespec="microseconds")} GPS"
          f"\n\t\t\t\tSat time "
          f"{sat_gps_time.isoformat(sep=" ", timespec="microseconds")} GPS")

    ### HOLY DOGSHIT WUT WAIT WUT JESUS CHRIST

    """
        Preamble found at: 
		    Sim time 2026-06-01 00:00:06.071414 GPS
	
	    Okay. Doesn't seem exciting
	    
	    UNTIL you realize 0.71414 is the actual satellite pseudorange in seconds
	    
	    DEBUG:gps_ranging:SVN 02 pseudorange: 21,409,507.60 m = 21,409,453.33 (actual) +  
	        50.96 (ionospheric) +  3.30 (tropospheric)
	        
	    21,409,507.6 / speed of light is 0.71414 seconds
	    
	    Once we read out the preamble and then back calculate to the time of the start of the preamble,
	        that gives us 6.071414 seconds
	        
	    Take the fractional second out of the preamble and somehow we have a pseudorange
	    
	    This is black magic of the highest sort
	"""


def _add_sat_clock_to_sim(sim_engine: sim_time.SimTime, prn_num: int) -> None:
    ota_delay: float = _hidden_pseudoranges[prn_num] / SPEED_OF_LIGHT_M_PER_S

    sv_time_absolute_gps: datetime.datetime = sim_engine.user_clock("receiver_time_absolute_gps").time_current() - \
        datetime.timedelta(seconds=ota_delay)

    sim_engine.add_user_clock(
        sim_time.SimTime.UserClock(f"prn{prn_num:02d}_absolute_gps", sv_time_absolute_gps,
                                   logging_level=logging.DEBUG
        )
    )


def _establish_prn_sync(args: argparse.Namespace, sim_engine: sim_time.SimTime, prn_num: int,
                        fix_svn_prns: dict[int, numpy.typing.NDArray[numpy.int8]]) -> float:
    _logger.debug(f"Starting PRN sync establishment for SV with PRN {prn_num:02d}")
    time_for_prn_sync_seconds: float = 0.001
    sim_engine.advance_sim_time(datetime.timedelta(seconds=time_for_prn_sync_seconds))

    pseudorange_in_bits_per_svn: dict[int, int] = {}
    pseudorange_in_bits: int = _pseudorange_by_received_prn_stream_bits(args, prn_num, fix_svn_prns[prn_num])
    pseudorange_in_bits_per_svn[prn_num] = pseudorange_in_bits

    sim_prn_start_offset: float = pseudorange_in_bits_per_svn[prn_num] / CA_CODE_CHIPPING_RATE
    print(f"\t\t\tSub-millisecond portion of pseudorange to SV with PRN {prn_num:02} : "
          f"{sim_prn_start_offset:.06f} s")

    print("\t\t\tSim time elapsed at moment PRN sync is established: "
          f"{sim_time.SimTime.timedelta_isoformat(sim_engine.user_clock(
              "receiver_time_absolute_gps").time_elapsed())} s")

    _logger.info(f"PRN sync successfully established for SV with PRN {prn_num:02d}, "
                f"time offset = {sim_prn_start_offset:.06f} s")

    return sim_prn_start_offset


def _main() -> None:
    global simulation_current_gps_time_at_receiver

    args: argparse.Namespace = _parse_args()

    # Populate the gold codes for the satellites we're simulating a lock to
    print()
    print("Generating GPS Gold codes for satellites we are going to use in our position fix")
    gold_codes: gps_gold_codes.GPSGoldCodes = gps_gold_codes.GPSGoldCodes()
    fix_svn_prns: dict[int, numpy.typing.NDArray[numpy.int8]] = {}
    for prn_num in (2, 7, 13, 19):
        fix_svn_prns[prn_num] = gold_codes.gold_code(prn_num)
    print("\tDone!")

    # Create sim engine with all user clocks
    sim_engine: sim_time.SimTime = sim_time.SimTime(
        logging_level=logging.DEBUG
    )
    receiver_time_absolute_gps: datetime.datetime = datetime.datetime.fromisoformat("2026-06-01T00:00:01.000000")
    sim_engine.add_user_clock(
        sim_time.SimTime.UserClock("receiver_time_absolute_gps", receiver_time_absolute_gps,
                                   logging_level=logging.DEBUG
        )
    )


    print()
    print("Calculating pseudoranges for all satellites in the fix")

    for prn_num in [2, 7, 13, 19]:

        print()
        print(f"\tStarting processing for SV with PRN {prn_num:02}")

        _determine_sat_hidden_pseudorange(args, prn_num)
        _add_sat_clock_to_sim(sim_engine, prn_num)

        print()
        print(f"\t\tStep 01: PRN Gold code sync")
        prn_sync_offset: float = _establish_prn_sync(args, sim_engine, prn_num, fix_svn_prns)

        # Now we have locked sync on the most fundamental signal, now establish bit sync to LNAV message
        #   data stream so we can start to read LNAV messages.
        #
        # We needed C/PRN as a prereq as it tells us precise time windows to watch when LNAV data bit flips *may*
        #   happen (every 20 repetitions of full C/A PRN, e.g. 0.020 s)
        print()
        print(f"\t\tStep 02: bit sync")
        _establish_bit_sync(args, sim_engine, prn_num, prn_sync_offset)

        sim_time_offsets_lnav_bit_starts: list[float] = _get_sim_time_offsets_lnav_bit_starts()

        print(f"\t\t\tSim time sub-second offsets of LNAV bit starts: {sim_time_offsets_lnav_bit_starts[0]:8.06f} s "
              "+ integer multiples of 0.020 s (20 ms)")

        raise NotImplementedError("Stop")


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
