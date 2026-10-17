"""Seeded random streams, one per agent.

Reproducibility rule: no code in this project draws from a global random generator. Every random
number comes from a stream created here, from the run seed. A stream depends *only* on the seed and
the agent id, never on how many other agents exist or in which order they were created. So adding a
second market maker does not change a single number the noise traders draw, which makes it possible
to change one thing in an experiment and see only that change.
"""

from __future__ import annotations

from numpy.random import PCG64, Generator, SeedSequence


class AgentStreams:
    """Hands out one independent ``numpy`` generator per agent id, all derived from one seed."""

    def __init__(self, seed: int) -> None:
        """Create the stream factory for a run. ``seed`` must be a non-negative integer."""
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        self.seed = seed
        self._streams: dict[int, Generator] = {}

    def for_agent(self, agent_id: int) -> Generator:
        """Return the agent's generator; repeated calls return the same, advancing stream."""
        if isinstance(agent_id, bool) or not isinstance(agent_id, int) or agent_id < 0:
            raise ValueError(f"agent_id must be a non-negative integer, got {agent_id!r}")
        stream = self._streams.get(agent_id)
        if stream is None:
            # spawn_key is how numpy derives independent child streams from one parent seed.
            stream = Generator(PCG64(SeedSequence(self.seed, spawn_key=(agent_id,))))
            self._streams[agent_id] = stream
        return stream
