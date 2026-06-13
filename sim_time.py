import datetime
import logging
import typing


def timedelta_str(time_delta: datetime.timedelta) -> str:
    return f"{time_delta.total_seconds():.06f}"


class SimTimer:
    def __init__(self: typing.Self,
                 timer_name: str,
                 timer_rollover_microseconds: int | None = None,
                 logging_level: int = logging.WARNING) -> None:

        self.timer_name: str = timer_name
        self._time_elapsed: datetime.timedelta = datetime.timedelta()
        self._timer_rollover_microseconds: int | None = timer_rollover_microseconds
        self._logger: logging.Logger = logging.getLogger(f"sim_time.timer.{timer_name}")
        self._logger.setLevel(logging_level)


    def advance_time(self: typing.Self, advance_amount: datetime.timedelta) -> None:
        self._time_elapsed += advance_amount

        # Handle possible rollover
        if self._timer_rollover_microseconds is not None:
            timer_in_microseconds: int = int(self._time_elapsed.total_seconds() * 1_000_000)
            timer_with_rollover: int = timer_in_microseconds % self._timer_rollover_microseconds

            if timer_in_microseconds != timer_with_rollover:
                self._logger.info("Time advance crossed rollover threshold")
                self._time_elapsed = datetime.timedelta(microseconds=timer_with_rollover)

        self._logger.debug(f"Timer advanced by {timedelta_str(advance_amount)} s, "
                           f"now at {self._time_elapsed.total_seconds():.06f} s")


    def value(self: typing.Self) -> datetime.time:
        return (datetime.datetime.min + self._time_elapsed).time()


class SimClock:

    def __init__(self: typing.Self,
                 clock_name: str,
                 initial_clock: datetime.datetime | datetime.time,
                 logging_level: int = logging.WARNING) -> None:

        self.clock_name: str = clock_name
        self._time_initial: datetime.datetime

        if isinstance(initial_clock, datetime.datetime):
            self._time_initial = initial_clock
        elif isinstance(initial_clock, datetime.time):
            self._time_initial = datetime.datetime.combine(
                datetime.date.fromisoformat("1970-01-01"),
                initial_clock
            )
        else:
            raise ValueError("Initial clock must be of type datetime.datetime or datetime.time")

        self._time_current: datetime.datetime = self._time_initial
        self._logger = logging.getLogger(f"sim_time.clock.{clock_name}")
        self._logger.setLevel(logging_level)


    def advance_time(self: typing.Self, advance_amount: datetime.timedelta) -> None:
        self._time_current += advance_amount
        self._logger.debug(f"Clock advanced by {timedelta_str(advance_amount)} s, "
                           f"now at {self._time_current.isoformat(sep=" ", timespec="microseconds")}")


    def time_current(self: typing.Self) -> datetime.datetime:
        return self._time_current


    def time_elapsed(self: typing.Self) -> datetime.timedelta:
        return self._time_current - self._time_initial


class SimTime:

    # No need to call this constructor on every advance time call, set it and forget it
    _zero_delta: datetime.timedelta = datetime.timedelta()

    # Class level logger
    _logger: logging.Logger = logging.getLogger("sim_time")


    def __init__(self: typing.Self, logging_level: int = logging.WARNING) -> None:
        self._sim_timer: SimTimer = SimTimer("sim_master_time")
        self._user_clocks: dict[str, SimClock] = {}
        self._user_timers: dict[str, SimTimer] = {}
        self._logger.setLevel(logging_level)


    def sim_master_time(self: typing.Self) -> datetime.time:
        return self._sim_timer.value()


    def add_user_clock(self: typing.Self, user_clock: SimClock) -> None:
        if user_clock.clock_name in self._user_clocks:
            raise ValueError(f"Tried to add clock named {user_clock.clock_name} that was already added")

        self._user_clocks[user_clock.clock_name] = user_clock
        SimTime._logger.info(f"Added user clock \"{user_clock.clock_name}\" with initial value of "
                             f"{user_clock.time_current().isoformat(sep=" ", timespec="microseconds")}")


    def add_user_timer(self: typing.Self, timer: SimTimer) -> None:
        if timer.timer_name in self._user_timers:
            raise ValueError(f"Tried to add timer named {timer.timer_name} that was already added")

        self._user_timers[timer.timer_name] = timer
        SimTime._logger.info(f"Added user timer \"{timer.timer_name}\" with initial value of "
                             f"{self._user_timers[timer.timer_name].value().isoformat(timespec="microseconds")}")


    def advance_sim_time(self: typing.Self, time_shift: datetime.timedelta | float) -> None:
        time_delta: datetime.timedelta

        if isinstance(time_shift, float):
            time_delta = datetime.timedelta(seconds=time_shift)
        elif isinstance(time_shift, datetime.timedelta):
            time_delta = time_shift
        else:
            raise ValueError("Shift must be of type datetime.timedelta or float")

        SimTime._logger.debug(f"Sim engine starting time advance of {timedelta_str(time_delta)} s")

        if time_delta < SimTime._zero_delta:
            raise ValueError("Tried to move time backwards")

        if time_delta == SimTime._zero_delta:
            # No-op
            return

        self._sim_timer.advance_time(time_delta)

        # Now add the delta to all individual clocks and timers
        for curr_clock in self._user_clocks.values():
            curr_clock.advance_time(time_delta)
        for curr_timer in self._user_timers.values():
            curr_timer.advance_time(time_delta)

        SimTime._logger.info(f"Sim engine completed time advance of {timedelta_str(time_delta)} s")


    def user_clock(self: typing.Self, clock_name: str) -> SimClock:
        if clock_name not in self._user_clocks:
            raise ValueError(f"Requested non-existent clock \"{clock_name}\"")

        return self._user_clocks[clock_name]


    def user_clocks(self: typing.Self) -> dict[str, SimClock]:
        return self._user_clocks


    def user_timer(self: typing.Self, name: str) -> SimTimer:
        if name not in self._user_timers:
            raise ValueError(f"Requested non-existent timer \"{name}\"")

        return self._user_timers[name]


    def user_timers(self: typing.Self) -> dict[str, SimTimer]:
        return self._user_timers


if __name__ == "__main__":

    def _print_tickers(clocks_to_print: dict[str, SimClock], timers_to_print: dict[str, SimTimer]):
        print("\tUser Clocks:")
        for clock_name, user_clock in sorted(clocks_to_print.items()):
            current_datetime: datetime.datetime = user_clock.time_current()
            print(f"\t\t{clock_name:26} : "
                  f"{current_datetime.isoformat(sep=" ", timespec="microseconds")}")

        print("\tUser Timers:")
        for name, timer in sorted(timers_to_print.items()):
            print(f"\t\t{name:26} :            "
                  f"{timer.value().isoformat(timespec="microseconds")}")

    logging.basicConfig()

    sim_engine: SimTime = SimTime(
        # logging_level=logging.DEBUG
    )

    receiver_time_absolute_gps: datetime.datetime = datetime.datetime.fromisoformat("2026-06-01T00:00:01.000000")
    sim_engine.add_user_clock(
        SimClock("receiver_time_absolute_gps", receiver_time_absolute_gps,
            # logging_level=logging.DEBUG
        )
    )

    sat_time_absolute_gps: datetime.datetime = receiver_time_absolute_gps - datetime.timedelta(seconds=0.071414)
    sim_engine.add_user_clock(
        SimClock("sat_time_absolute_gps", sat_time_absolute_gps,
            # logging_level=logging.DEBUG
        )
    )

    print()
    print("Tickers at starting line:")
    _print_tickers(sim_engine.user_clocks(), sim_engine.user_timers())

    time_to_advance: datetime.timedelta = datetime.timedelta(seconds=5.071414)
    sim_engine.advance_sim_time(time_to_advance)
    print(f"\nAdvanced sim time by {timedelta_str(time_to_advance)} s")

    print()
    print("Tickers after time advance:")
    _print_tickers(sim_engine.user_clocks(), sim_engine.user_timers())

    print()
    print("Elapsed time reported by all user clocks:")
    for clock_name, user_clock in sorted(sim_engine.user_clocks().items()):
        print(f"\t{clock_name:26} : "
              f"{timedelta_str(user_clock.time_elapsed()):>13} s")
