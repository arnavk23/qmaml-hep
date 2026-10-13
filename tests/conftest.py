"""Shared test configuration."""

from hypothesis import HealthCheck, settings

# Property tests generate small arrays; on a loaded CI runner or under coverage, generation can trip Hypothesis's
# "too slow" health check, which says nothing about correctness.
settings.register_profile("default", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("default")
