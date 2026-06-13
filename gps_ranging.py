import argparse
import datetime
import json
import logging
import math
import random

import numpy
import numpy.linalg
import numpy.typing

import gps_gold_codes
import gps_ephemeris
import gps_receiver
import sim_time

_logger = logging.getLogger("gps_ranging")
_logger.setLevel(logging.WARNING)


SPEED_OF_LIGHT_M_PER_S: float = 299_792_458.0
PRN_RATE_BITS_PER_MS: int = 1_023
CA_CODE_CHIPPING_RATE: int = 1_023 * 1_000
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
    bit_sync_operation_start: datetime.timedelta = datetime.timedelta(
        seconds=sim_engine.sim_master_time().second,
        microseconds=sim_engine.sim_master_time().microsecond,
    )

    print("\t\t\tUsing 1 ms timer created during PRN sync to signal exact time to pop up and listen to the modulated"
          "\n\t\t\t\tnav signal for a transition edge marking the ms when nav message bits start")

    satellite_pseudorange_meters: float = _hidden_pseudoranges[prn_num]

    # Find out where we are in the bitstream based on full transmission delay
    full_over_the_air_transmission_delay_seconds: float = _hidden_pseudoranges[prn_num] / SPEED_OF_LIGHT_M_PER_S

    transmission_delay_offset_chips: int = math.ceil( (1_023 * 1_000) * full_over_the_air_transmission_delay_seconds)

    # Compute sat delta to next 20ms interval (time instants when LNAV bits start)
    sv_clock: sim_time.SimClock = sim_engine.user_clock(f"prn{prn_num:02d}_absolute_gps")
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
        f"Bit sync: advancing clock {sim_time.timedelta_str(delay_to_next_bit_transition_edge)} s "
         "as it's when bit sync is established (receiver sees its first (0 <-> 1) transition edge at a data bit boundary"
    )
    sim_engine.advance_sim_time(delay_to_next_bit_transition_edge)

    # Receiver relative time when bit sync starts
    bit_sync_operation_end: datetime.timedelta = datetime.timedelta(
        seconds=sim_engine.sim_master_time().second,
        microseconds=sim_engine.sim_master_time().microsecond,
    )

    cumulative_time_to_establish_bit_sync: datetime.timedelta = bit_sync_operation_end - bit_sync_operation_start

    print( "\t\t\tEstablishing bit sync took "
          f"{sim_time.timedelta_str(cumulative_time_to_establish_bit_sync)} s")

    print("\t\t\tStarting 20 ms timer to signal when every new nav message bit first hits the receiver antenna")

    print("\t\t\tTime from receiver cold start through established bit sync: "
          f"{sim_engine.sim_master_time().isoformat(timespec="microseconds")} s")


def _add_sat_clock_to_sim(sim_engine: sim_time.SimTime, prn_num: int) -> None:
    sim_master_time: datetime.time = sim_engine.sim_master_time()

    ota_time: datetime.timedelta = datetime.timedelta(
        seconds=_hidden_pseudoranges[prn_num] / SPEED_OF_LIGHT_M_PER_S
    )

    # Find GPS time + the offset for the sim
    sv_clock_gps: datetime.datetime = datetime.datetime(
        year=2026,
        month=6,
        day=1,
        hour=sim_master_time.hour,
        minute=sim_master_time.minute,
        second=1 + sim_master_time.second,
        microsecond=sim_master_time.microsecond
    ) - ota_time


    sim_engine.add_user_clock(
        sim_time.SimClock(f"prn{prn_num:02d}_absolute_gps", sv_clock_gps,
                          # logging_level=logging.DEBUG
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

    print(f"\t\t\tEstablishing PRN sync took {sim_engine.sim_master_time().isoformat(timespec="microseconds")} s")
    print( "\t\t\tStarting 1 ms timer to signal each precise time the PRN Gold code for this SV restarts at the antenna")

    _logger.info(f"PRN sync successfully established for SV with PRN {prn_num:02d}")
    _logger.info(f"1 ms timer set which fires every time the antenna receives the start of a new PRN loop")

    return sim_prn_start_offset


def _next_word_boundary_time(current_time: datetime.datetime) -> datetime.datetime:
    # 1. Get the start of the current minute
    start_of_minute:datetime.datetime = current_time.replace(second=0, microsecond=0)

    # 2. Convert elapsed time in this minute into microseconds
    elapsed_us = int((current_time - start_of_minute).total_seconds() * 1_000_000)

    # 3. Define the interval in microseconds (600 ms)
    interval_us = 600_000

    # 4. Find the next interval boundary
    intervals_passed = elapsed_us // interval_us
    next_interval_us = (intervals_passed + 1) * interval_us

    # 5. Add back to the start of the minute to get the final timestamp
    return start_of_minute + datetime.timedelta(microseconds=next_interval_us)


def _establish_word_sync(sim_engine: sim_time.SimTime, prn_num: int) -> None:

    # According to the Googles, you can establish word sync after listening to two full words and
    # checking parity within those two thirty bit words
    _logger.debug(f"Starting word sync establishment for SV with PRN {prn_num:02d}")

    # Receiver relative time when bit sync starts
    sync_operation_start: datetime.timedelta = datetime.timedelta(
        seconds=sim_engine.sim_master_time().second,
        microseconds=sim_engine.sim_master_time().microsecond,
    )


    sv_abs_time: datetime.datetime = sim_engine.user_clock(f"prn{prn_num:02d}_absolute_gps").time_current()

    # Find next time we'll send a 0.600000 second word
    next_word_boundary: datetime.datetime = _next_word_boundary_time(sv_abs_time)

    _logger.debug(f"Current SV time: {sv_abs_time.isoformat(sep=" ", timespec="microseconds")} GPS")
    _logger.debug( "SV time when next subframe will be sent: "
                  f"{next_word_boundary.isoformat(sep=" ", timespec="microseconds")} GPS")

    delta_to_next_word_boundary: datetime.timedelta = next_word_boundary - sv_abs_time

    _logger.info( f"Advancing clock {sim_time.timedelta_str(delta_to_next_word_boundary)} s to next word start" )
    sim_engine.advance_sim_time(delta_to_next_word_boundary)

    # Read two full words at which time we'll have word sync
    two_word_duration: datetime.timedelta = datetime.timedelta(seconds=2 * BITS_PER_LNAV_WORD *
                                                                       SECONDS_PER_LNAV_DATA_BIT)
    _logger.info( f"Advancing clock {sim_time.timedelta_str(two_word_duration)} s to read two full words"
                   ", at which time we've hit word sync")
    sim_engine.advance_sim_time(two_word_duration)

    # Receiver relative time when bit sync starts
    sync_operation_end: datetime.timedelta = datetime.timedelta(
        seconds=sim_engine.sim_master_time().second,
        microseconds=sim_engine.sim_master_time().microsecond,
    )

    cumulative_time_to_establish_sync: datetime.timedelta = sync_operation_end - sync_operation_start

    print( "\t\t\tEstablishing word sync took "
          f"{sim_time.timedelta_str(cumulative_time_to_establish_sync)} s")

    print("\t\t\tTime from receiver cold start through established word sync: "
          f"{sim_engine.sim_master_time().isoformat(timespec="microseconds")} s")



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

    print()
    print("Calculating pseudoranges for all satellites in the fix")

    for prn_num in [2, 7, 13, 19]:

        # Create sim engine with all user clocks
        sim_engine: sim_time.SimTime = sim_time.SimTime(
            logging_level=logging.WARNING
        )

        print()
        print(f"\tStarting processing for SV with PRN {prn_num:02}")

        _determine_sat_hidden_pseudorange(args, prn_num)
        _add_sat_clock_to_sim(sim_engine, prn_num)

        # Timer that will give us our pseudorange to this SV in seconds
        print()
        print(f"\t\tStep 1: start pseudorange timer")
        sim_engine.add_user_timer(sim_time.SimTimer(f"prn{prn_num:02}_pseudorange",
                                                    timer_rollover_microseconds=100_000,
                                                    logging_level=logging.WARN))
        print("\t\t\tDone!")

        print()
        print(f"\t\tStep 2: PRN Gold code sync (1 ms period)")
        _establish_prn_sync(args, sim_engine, prn_num, fix_svn_prns)

        # Now we have locked sync on the most fundamental signal, now establish bit sync with LNAV message
        #   data stream so we can start to read LNAV messages.
        #
        # We needed PRN sync as a prereq as that 1 ms timer tells the receiver precise time windows to watch
        #   when LNAV data bit flips *may* happen (every 20 ms on PRN loop boundaries)
        print()
        print(f"\t\tStep 3: bit sync (20 ms period)")
        _establish_bit_sync(sim_engine, prn_num)

        # Now we have locked *both* PRN sync AND bit sync, we can start to watch bits to get word lock.
        #   Words are 30 bit long. 30 bits @ 20ms/bit = 600 ms/word.
        print()
        print(f"\t\tStep 4: word sync (600 ms period)")
        _establish_word_sync(sim_engine, prn_num)

        print()
        print("\t\tStep 5: set pseudorange to SV")
        sv_pseudorange_in_seconds: float = sim_engine.user_timer(
            f"prn{prn_num:02}_pseudorange").value().microsecond / 1_000_000

        print( "\t\t\tValue of pseudorange timer started at step 1: "
              f"{sv_pseudorange_in_seconds:8.06f} s")
        sv_pseudorange_in_meters: float = sv_pseudorange_in_seconds * SPEED_OF_LIGHT_M_PER_S
        print(f"\t\t\tCalculated pseudorange to SV        : {sv_pseudorange_in_meters:12,.01f} m")

        print(f"\t\t\tHidden pseudorange used for the sim : {_hidden_pseudoranges[prn_num]:12,.01f} m")

    print()


if __name__ == "__main__":
    logging.basicConfig()
    _main()
