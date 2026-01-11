import sys
import types
import sqlite3
from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np
import pytest

# Utility to load the target module from a non-standard filename
MODULE_PATH = Path(__file__).resolve().parents[1] / "1_Migration_to_SQL"

def load_module_with_dummy_fred():
    # Inject a dummy fredapi module if it's not installed
    if 'fredapi' not in sys.modules:
        dummy = types.ModuleType('fredapi')
        class DummyFred:
            def __init__(self, api_key=None):
                self.api_key = api_key
            def get_series(self, code):
                # default empty series, tests override as needed
                return pd.Series(dtype=float)
        dummy.Fred = DummyFred
        sys.modules['fredapi'] = dummy

    spec = importlib.util.spec_from_file_location("financial_migrator", str(MODULE_PATH))
    mod = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(mod)
    return mod


@pytest.fixture
def migrator():
    mod = load_module_with_dummy_fred()
    m = mod.FinancialDataMigrator(db_path=':memory:')
    try:
        yield m
    finally:
        m.close()


def test_create_tables(migrator):
    cur = migrator.conn.cursor()
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        'stock_prices', 'stock_metadata', 'financial_statements', 'economic_indicators', 'market_indices'
    }.issubset(tables)


def test_migrate_stock_csv_ticker_inserts_prices(tmp_path, migrator):
    # Prepare a small CSV
    df = pd.DataFrame({
        'Date': pd.to_datetime(['2020-01-01', '2020-02-01']),
        'Open': [10.0, 12.0],
        'High': [11.0, 13.0],
        'Low': [9.0, 11.5],
        'Close': [10.5, 12.5],
        'Adj Close': [10.4, 12.4],
        'Volume': [1000, 2000],
    })
    csv_path = tmp_path / 'prices.csv'
    df.to_csv(csv_path, index=False)

    count = migrator.migrate_stock_csv(str(csv_path), ticker='TEST')
    assert count == len(df)

    res = pd.read_sql_query("SELECT COUNT(*) FROM stock_prices WHERE ticker='TEST'", migrator.conn)
    assert int(res.iloc[0, 0]) == len(df)


def test_migrate_stock_csv_index_inserts_indices(tmp_path, migrator):
    df = pd.DataFrame({
        'Date': pd.to_datetime(['2020-01-01', '2020-02-01']),
        'Close': [100.0, 101.0],
        'Adj Close': [99.5, 100.5],
        'Volume': [1_000_000, 1_100_000],
    })
    csv_path = tmp_path / 'index.csv'
    df.to_csv(csv_path, index=False)

    count = migrator.migrate_stock_csv(str(csv_path), index_name='MY_INDEX')
    assert count == len(df)

    res = pd.read_sql_query("SELECT COUNT(*) FROM market_indices WHERE index_name='MY_INDEX'", migrator.conn)
    assert int(res.iloc[0, 0]) == len(df)


def test_migrate_stock_excel_inserts_financials(monkeypatch, migrator):
    # Mock pd.read_excel to return the dict of DataFrames as the code expects
    income = pd.DataFrame({
        'calendarYear': [2019, 2020],
        'Revenue': [100.0, 110.0],
        'COGS': [40.0, 45.0],
    }).set_index(pd.Index(['calendarYear', 'Revenue', 'COGS']))

    balance = pd.DataFrame({
        'calendarYear': [2019, 2020],
        'Assets': [500.0, 520.0],
        'Liabilities': [200.0, 210.0],
    }).set_index(pd.Index(['calendarYear', 'Assets', 'Liabilities']))

    cashflow = pd.DataFrame({
        'calendarYear': [2019, 2020],
        'Operating CF': [60.0, 65.0],
        'CapEx': [-20.0, -22.0],
    }).set_index(pd.Index(['calendarYear', 'Operating CF', 'CapEx']))

    excel_return = {
        'Income Statement': income,
        'Balance Sheet': balance,
        'Cash Flow': cashflow,
    }

    monkeypatch.setattr(pd, 'read_excel', lambda *args, **kwargs: excel_return)

    inserted = migrator.migrate_stock_excel('dummy.xlsx', ticker='XYZ', data_type='FY')

    # Expect number of metrics (rows excluding calendarYear) * number of years across 3 statements
    n_years = 2
    n_metrics = (2 + 2 + 2)  # Revenue, COGS, Assets, Liabilities, Operating CF, CapEx
    assert inserted == n_years * n_metrics

    res = pd.read_sql_query(
        "SELECT COUNT(*) FROM financial_statements WHERE ticker='XYZ'", migrator.conn
    )
    assert int(res.iloc[0, 0]) == n_years * n_metrics


def test_migrate_fred_without_key_returns_zero(capsys):
    mod = load_module_with_dummy_fred()
    m = mod.FinancialDataMigrator(db_path=':memory:')
    try:
        count = m.migrate_fred_data([{'code': 'UNRATE', 'name': 'Unemployment Rate'}])
        captured = capsys.readouterr()
        assert count == 0
        assert 'FRED API key not provided' in captured.out
    finally:
        m.close()


def test_migrate_fred_with_mocked_series(monkeypatch):
    mod = load_module_with_dummy_fred()
    m = mod.FinancialDataMigrator(db_path=':memory:', fred_api_key='dummy')

    try:
        # Mock get_series to return a simple series
        dates = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        series = pd.Series([3.5, np.nan, 4.0], index=dates)

        def fake_get_series(code):
            assert code == 'UNRATE'
            return series

        monkeypatch.setattr(m.fred, 'get_series', fake_get_series)
        monkeypatch.setattr(mod.time, 'sleep', lambda x: None)

        count = m.migrate_fred_data([
            {'code': 'UNRATE', 'name': 'Unemployment Rate', 'category': 'labor', 'frequency': 'monthly'}
        ])

        assert count == 2  # one NaN filtered out
        res = pd.read_sql_query(
            "SELECT COUNT(*) FROM economic_indicators WHERE indicator_code='UNRATE'", m.conn
        )
        assert int(res.iloc[0, 0]) == 2
    finally:
        m.close()


def test_query_stock_prices_filters_by_date(tmp_path, migrator):
    # Insert some data directly
    rows = [
        ('AAA', '2020-01-01', 10, 11, 9, 10.5, 10.5, 1000, 'CSV'),
        ('AAA', '2020-02-01', 11, 12, 10, 11.5, 11.5, 1100, 'CSV'),
        ('AAA', '2020-03-01', 12, 13, 11, 12.5, 12.5, 1200, 'CSV'),
    ]
    migrator.conn.executemany(
        """
        INSERT INTO stock_prices (ticker, date, open, high, low, close, adj_close, volume, data_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    migrator.conn.commit()

    df = migrator.query_stock_prices('AAA', start_date='2020-02-01', end_date='2020-03-01')
    assert list(pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')) == ['2020-02-01', '2020-03-01']


def test_add_stock_metadata_upsert(migrator):
    migrator.add_stock_metadata('BBB', company_name='Beta Inc.', sector='Tech')
    df1 = pd.read_sql_query("SELECT company_name, sector FROM stock_metadata WHERE ticker='BBB'", migrator.conn)
    assert df1.iloc[0]['company_name'] == 'Beta Inc.'

    migrator.add_stock_metadata('BBB', company_name='Beta Corp.', sector='Technology')
    df2 = pd.read_sql_query("SELECT company_name, sector FROM stock_metadata WHERE ticker='BBB'", migrator.conn)
    assert df2.iloc[0]['company_name'] == 'Beta Corp.'
    assert df2.iloc[0]['sector'] == 'Technology'
