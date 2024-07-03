import pytest

from tests.fixtures import init_simulation_casestudy_api


def test_interactive_mode():
    sim = init_simulation_casestudy_api()
    sim.interactive()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_scripting_API()
    # test_interactive_mode()
