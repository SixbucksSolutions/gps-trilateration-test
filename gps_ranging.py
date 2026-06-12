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

SECONDS_PER_LNAV_DATA_BIT: float = 0.020
BITS_PER_LNAV_WORD: int = 30

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


def _establish_bit_sync(sim_engine: sim_time.SimTime, prn_num: int) -> None:
    _logger.debug(f"Starting to establish bit sync")

    # Receiver relative time when bit sync starts
    bit_sync_operation_start: datetime.timedelta = sim_engine.user_clock("receiver_time_absolute_gps").time_elapsed()

    print("\t\t\tUsing 1 ms timer created during PRN sync to signal exact time to pop up and listen to the modulated"
          "\n\t\t\t\tnav signal for a transition edge marking the ms when nav message bits start")

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

    # Advance sim time to first data bit transition edge we see in this sim run at which time we have
    #   achieved bit sync
    _logger.info(
        f"Bit sync: advancing clock {sim_time.SimTime.timedelta_isoformat(delay_to_next_bit_transition_edge)} s "
         "as it's when bit sync is established (receiver sees its first (0 <-> 1) transition edge at a data bit boundary"
    )
    sim_engine.advance_sim_time(delay_to_next_bit_transition_edge)

    # Receiver relative time when bit sync starts
    bit_sync_operation_end: datetime.timedelta = sim_engine.user_clock("receiver_time_absolute_gps").time_elapsed()

    cumulative_time_to_establish_bit_sync: datetime.timedelta = bit_sync_operation_end - bit_sync_operation_start

    print( "\t\t\tEstablishing bit sync took "
          f"{sim_time.SimTime.timedelta_isoformat(cumulative_time_to_establish_bit_sync)} s")

    print("\t\t\tStarting 20 ms timer to signal when every new nav message bit first hits the receiver antenna")

    print("\t\t\tTime from receiver cold start through established bit sync: "
          f"{sim_time.SimTime.timedelta_isoformat(sim_engine.user_clock(
              "receiver_time_absolute_gps").time_elapsed())} s")


def _get_tow_in_gps_seconds(time_instant_in_gps_time_reference: datetime.datetime) -> int:
    # Find days passed since the most recent Sunday
    # (.weekday() returns Mon=0, Tue=1 ... Sat=5, Sun=6)
    days_since_sunday = (time_instant_in_gps_time_reference.weekday() + 1) % 7

    # Create datetime of last Sunday at 00:00:00.000000 GPS time
    last_sunday_midnight = time_instant_in_gps_time_reference.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - datetime.timedelta(days=days_since_sunday)

    # Get the Time Of Week (TOW) which is time difference in whole seconds
    time_of_week: datetime.timedelta = time_instant_in_gps_time_reference - last_sunday_midnight

    if time_of_week.microseconds != 0:
        raise ValueError("Can only compute TOW for time instants on exact second boundaries")

    return int(time_of_week.total_seconds())


def _establish_subframe_sync(sim_engine: sim_time.SimTime, prn_num: int) -> int:
    """
    Simulate the receiver establishing subframe sync between the receiver and the SV with the specified PRN.

    :param sim_engine:
    :param prn_num:
    :return: the exact *GPS* time of week (in seconds) *at the sim's SV GPS time reference*
    """

    _logger.debug(f"Starting subframe sync with SV with PRN {prn_num:02d}")

    # Receiver relative time when bit sync starts
    sync_operation_start: datetime.timedelta = sim_engine.user_clock("receiver_time_absolute_gps").time_elapsed()


    print("\t\t\tUsing 20 ms timer created during bit sync to know when each bit will start, allowing the receiver to"
          "\n\t\t\t\tsave power, only listening at the exact 20 ms intervals when bits hit our receiver antenna")
    print("\t\t\tListening to the nav message bit stream for the first time it sees the fixed LNAV preamble bits")


    # Compute *SV* reference frame time delta to next 6-second interval (subframe starts)
    sv_clock: sim_time.SimTime.UserClock = sim_engine.user_clock("prn02_absolute_gps")
    sv_abs_time: datetime.datetime = sv_clock.time_current()

    # Calculate delta needed to reach the next six-second interval
    interval_s: int = 6
    absolute_s = sv_abs_time.second + interval_s - (sv_abs_time.second % interval_s)

    # Update seconds and zero out microseconds -- subframes start on exact second boundaries *in SV time reference*
    gps_time_of_next_subframe_start = sv_abs_time.replace(second=absolute_s, microsecond=0)

    _logger.debug(f"Current SV time: {sv_abs_time.isoformat(sep=" ", timespec="milliseconds")} GPS")
    _logger.debug( "SV time when next subframe will be sent: "
                  f"{gps_time_of_next_subframe_start.isoformat(sep=" ", timespec="milliseconds")} GPS")

    bit_reading_duration_to_start_of_next_subframe: datetime.timedelta = gps_time_of_next_subframe_start - sv_abs_time

    _logger.info( "Advancing clock "
                 f"{bit_reading_duration_to_start_of_next_subframe.seconds}."
                 f"{bit_reading_duration_to_start_of_next_subframe.microseconds:06d} s to next subframe start" )
    sim_engine.advance_sim_time(bit_reading_duration_to_start_of_next_subframe)

    # We can compute the time of week for the current moment in time that would be stored in the TOW
    #   field in the frame we're now starting to process
    time_of_week_at_subframe_start: int = _get_tow_in_gps_seconds(sim_engine.user_clock(
        "prn02_absolute_gps").time_current())

    _logger.debug( "Current GPS time of week at this exact moment: "
                  f"{time_of_week_at_subframe_start // 6:,}.000000000 seconds")

    # TOW value that will be contained in the subframe we're about to rip open is the subframe index (relative to
    #   start of GPS week) of the SUBSEQUENT subframe
    subframe_tow_payload: int = (time_of_week_at_subframe_start // 6) + 1

    # Now need to wait to let receiver read LNAV preamble
    bits_in_lnav_preamble: int = 8
    duration_to_read_preamble: datetime.timedelta = datetime.timedelta(
        seconds=SECONDS_PER_LNAV_DATA_BIT * bits_in_lnav_preamble)
    _logger.info("Advancing clock "
                 f"{sim_time.SimTime.timedelta_isoformat(duration_to_read_preamble)} s to read LNAV preamble")
    sim_engine.advance_sim_time(duration_to_read_preamble)

    # Read handover word, which includes time of week
    duration_to_read_handover_word: datetime.timedelta = datetime.timedelta(
        seconds=SECONDS_PER_LNAV_DATA_BIT * BITS_PER_LNAV_WORD)
    _logger.info("Advancing clock "
                 f"{sim_time.SimTime.timedelta_isoformat(duration_to_read_handover_word)} s to read handover word")
    sim_engine.advance_sim_time(duration_to_read_handover_word)

    # Would now have the time of week (six second subframe counter that resets every Sunday at midnight GPS)
    _logger.info(f"Read GPS time of week from handover word, TOW value = {subframe_tow_payload:,}")
    exact_gps_time_of_week_second_at_subframe_sync: int = subframe_tow_payload * 6

    _logger.info( "TOW value signifies that the moment in time the first bit of the SUBSEQUENT subframe is received "
                 f"is *exactly* {exact_gps_time_of_week_second_at_subframe_sync:,}.000000000 GPS seconds from "
                  "the start of GPS week")

    # Now read out remainder of this subframe, which is when subframe sync is established
    sv_clock  = sim_engine.user_clock("prn02_absolute_gps")
    sv_abs_time = sv_clock.time_current()
    absolute_s = sv_abs_time.second + interval_s - (sv_abs_time.second % interval_s)

    # Update seconds and zero out microseconds -- subframes start on exact second boundaries *in SV time reference*
    gps_time_of_next_subframe_start = sv_abs_time.replace(second=absolute_s, microsecond=0)

    _logger.debug(f"Current SV time: {sv_abs_time.isoformat(sep=" ", timespec="milliseconds")} GPS")
    _logger.debug( "SV time when next subframe will be sent: "
                  f"{gps_time_of_next_subframe_start.isoformat(sep=" ", timespec="milliseconds")} GPS")

    bit_reading_duration_to_start_of_next_subframe: datetime.timedelta = gps_time_of_next_subframe_start - sv_abs_time

    _logger.info( "Advancing clock "
                 f"{bit_reading_duration_to_start_of_next_subframe.seconds}."
                 f"{bit_reading_duration_to_start_of_next_subframe.microseconds:06d} s to next subframe start when "
                  "the receiver hits subframe sync")
    sim_engine.advance_sim_time(bit_reading_duration_to_start_of_next_subframe)

    # Receiver relative time when sync is established
    sync_operation_end: datetime.timedelta = sim_engine.user_clock("receiver_time_absolute_gps").time_elapsed()

    cumulative_time_to_establish_sync: datetime.timedelta = sync_operation_end - sync_operation_start

    print( "\t\t\tEstablishing subframe sync took "
          f"{sim_time.SimTime.timedelta_isoformat(cumulative_time_to_establish_sync)} s")

    return exact_gps_time_of_week_second_at_subframe_sync


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

    _logger.info("Advancing time 1 ms to listen to complete PRN loop so we can find sync offset")
    time_for_prn_sync_seconds: float = 0.001
    sim_engine.advance_sim_time(datetime.timedelta(seconds=time_for_prn_sync_seconds))

    pseudorange_in_bits_per_svn: dict[int, int] = {}
    pseudorange_in_bits: int = _pseudorange_by_received_prn_stream_bits(args, prn_num, fix_svn_prns[prn_num])
    pseudorange_in_bits_per_svn[prn_num] = pseudorange_in_bits

    sim_prn_start_offset: float = pseudorange_in_bits_per_svn[prn_num] / CA_CODE_CHIPPING_RATE
    print(f"\t\t\tSub-millisecond portion of pseudorange to SV with PRN {prn_num:02} : "
          f"{sim_prn_start_offset:.06f} s")

    _logger.info(f"Advancing time by {sim_prn_start_offset:.06f} s which is the exact moment PRN sync is established")
    sim_engine.advance_sim_time(sim_prn_start_offset)

    print("\t\t\tEstablishing PRN sync took "
          f"{sim_time.SimTime.timedelta_isoformat(sim_engine.user_clock(
              "receiver_time_absolute_gps").time_elapsed())} s")
    print("\t\t\tStarting 1 ms timer to signal each precise time the PRN Gold code for this SV restarts at the antenna")

    _logger.info(f"PRN sync successfully established for SV with PRN {prn_num:02d}")
    _logger.info(f"1 ms timer set which fires every time the antenna receives the start of a new PRN loop")

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
        print(f"\t\tStep 01: PRN Gold code sync (1 ms period)")
        _establish_prn_sync(args, sim_engine, prn_num, fix_svn_prns)

        # Now we have locked sync on the most fundamental signal, now establish bit sync with LNAV message
        #   data stream so we can start to read LNAV messages.
        #
        # We needed PRN sync as a prereq as that 1 ms timer tells the receiver precise time windows to watch
        #   when LNAV data bit flips *may* happen (every 20 ms on PRN loop boundaries)
        print()
        print(f"\t\tStep 02: bit sync (20 ms period)")
        _establish_bit_sync(sim_engine, prn_num)

        # Now we have locked *both* PRN sync AND bit sync, we can start to watch bits to spot the subframe preamble
        print()
        print(f"\t\tStep 03: subframe sync (6 second period)")
        gps_time_of_week_seconds_at_sv: int = _establish_subframe_sync(sim_engine, prn_num)

        print( "\t\t\tGPS time of week at our SV at this *exact moment*: "
              f"{gps_time_of_week_seconds_at_sv:,}.000000000 seconds")

        print()
        print("\t\tCalculating pseudorange now that we have TOW + time since TOW + fractional offset")
        print("\t\t\tDone!")

        break

    print()


if __name__ == "__main__":
    logging.basicConfig()
    _main()
