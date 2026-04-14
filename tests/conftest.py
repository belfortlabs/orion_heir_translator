"""Shared test fixtures for the orion_heir translator tests."""

import pytest


class MockSchemeParams:
    """Mock scheme parameters for testing without Orion dependency."""

    ring_degree = 8192
    ciphertext_modulus_chain = [1099511627689, 1099511627553, 1099511627457]
    auxiliary_modulus_chain = [1099511627689]
    plaintext_modulus = 2**45
    log_scale = 45
    log_n = 13
    slots = 4096


@pytest.fixture
def scheme_params():
    return MockSchemeParams()
