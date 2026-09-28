import pytest

from alpha_bench.engine.evaluator import FieldCatalog
from alpha_bench.engine.market import generate
from alpha_bench.engine.simulate import OfflineEngine
from alpha_bench.rules.engine import Rules


@pytest.fixture(scope="session")
def market():
    return generate(n_stocks=150, n_days=420, warmup=260, seed=3)


@pytest.fixture(scope="session")
def catalog():
    return FieldCatalog()


@pytest.fixture(scope="session")
def engine(market, catalog):
    return OfflineEngine(market, catalog)


@pytest.fixture
def rules(tmp_path):
    return Rules(tmp_path)
