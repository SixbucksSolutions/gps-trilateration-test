import datetime
import json
import logging
import typing


class SimTime:

    class UserClock:

        def __init__(self: typing.Self,
                     clock_name: str,
                     initial_clock: datetime.datetime | datetime.time) -> None:

            self.clock_name: str = clock_name
            self.time_initial: datetime.datetime

            if isinstance(initial_clock, datetime.datetime):
                self.time_initial = initial_clock
            elif isinstance(initial_clock, datetime.time):
                self.time_initial = datetime.datetime.combine(
                    datetime.date.fromisoformat("1970-01-01"),
                    initial_clock
                )
            else:
                raise ValueError("Initial clock must be of type datetime.datetime or datetime.time")

            self.time_current: datetime.datetime = self.time_initial


    # No need to call this constructor on every advance time call, set it and forget it
    _zero_delta: datetime.timedelta = datetime.timedelta()

    # Class level logger
    _logger: logging.Logger = logging.getLogger("sim_time")
    _logger.setLevel(logging.DEBUG)


    @classmethod
    def _timedelta_isoformat(cls, time_delta: datetime.timedelta) -> str:
        total_seconds: int = time_delta.seconds
        microseconds: int = time_delta.microseconds

        return f"{total_seconds}.{microseconds:06d}"


    def __init__(self: typing.Self) -> None:
        self._sim_relative_clock: datetime.time = datetime.time()
        self._user_clocks: dict[str, SimTime.UserClock] = {}


    def add_user_clock(self: typing.Self, user_clock: UserClock) -> None:
        if user_clock.clock_name in self._user_clocks:
            raise ValueError(f"Tried to add clock named {user_clock.clock_name} that was already added")

        self._user_clocks[user_clock.clock_name] = user_clock
        SimTime._logger.info(f"Added user clock \"{user_clock.clock_name}\" at "
                             f"{user_clock.time_current.isoformat(sep=" ", timespec="microseconds")}")


    def advance_sim_time(self: typing.Self, time_delta: datetime.timedelta) -> None:
        if time_delta < SimTime._zero_delta:
            raise ValueError("Tried to move time backwards")

        if time_delta == SimTime._zero_delta:
            # No-op
            return

        self._sim_relative_clock += time_delta

        # Now add the delta to all individual clocks
        for curr_clock in self._user_clocks.values():
            curr_clock.time_current += time_delta

        SimTime._logger.info(f"Advanced simulation time by {SimTime._timedelta_isoformat(time_delta)} s")


    def user_clock(self: typing.Self, clock_name: str) -> datetime.datetime:
        if clock_name not in self._user_clocks:
            raise ValueError(f"Requested non-existent clock \"{clock_name}\"")

        return self.user_clocks()[clock_name]


    def user_clocks(self: typing.Self) -> dict[str, datetime.datetime]:
        curr_clock_times: dict[str, datetime.datetime] = {}

        for clock_name, user_clock in self._user_clocks.items():
            curr_clock_times[clock_name] = user_clock.time_current

        return curr_clock_times


if __name__ == "__main__":
    logging.basicConfig()

    sim_engine: SimTime = SimTime()
    sim_engine.add_user_clock(
        SimTime.UserClock("sim_time_relative", datetime.time())
    )

    receiver_time_absolute_gps: datetime.datetime = datetime.datetime.fromisoformat("2026-06-01T00:00:00.000000")
    sim_engine.add_user_clock(
        SimTime.UserClock("receiver_time_absolute_gps", receiver_time_absolute_gps)
    )

    sat_time_absolute_gps: datetime.datetime = receiver_time_absolute_gps - datetime.timedelta(seconds=0.071414)
    sim_engine.add_user_clock(
        SimTime.UserClock("sat_time_absolute_gps", sat_time_absolute_gps)
    )

    print("Clocks at starting line:")
    retrieved_clocks: dict[str, datetime.datetime] = sim_engine.user_clocks()

    for curr_clock_name in sorted(retrieved_clocks):
        print(f"{curr_clock_name}: {retrieved_clocks[curr_clock_name].isoformat(sep=" ", timespec="microseconds")}")
