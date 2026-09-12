import json
from pathlib import Path

import pandas as pd

try:
    from .yfinance_utils import download_ticker_data
except ImportError:
    from yfinance_utils import download_ticker_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "stock_metadata.json"
UNPLACED_DIR = DATA_DIR / "UnPlaced"


class AmbiguousTickerDataError(RuntimeError):
    """Raised when an unclassified ticker has multiple possible CSV files."""


def normalize_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        raise TypeError("Ticker symbol must be a string.")

    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("Ticker symbol must not be empty.")
    if (
        normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or "\x00" in normalized
        or Path(normalized).is_absolute()
    ):
        raise ValueError(f"Ticker symbol contains an unsafe path component: {ticker!r}")
    return normalized


def load_stock_metadata() -> dict:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Missing ticker metadata file: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        raw_metadata = json.load(file)

    if not isinstance(raw_metadata, dict):
        raise ValueError(f"Ticker metadata must be a JSON object: {METADATA_PATH}")

    metadata = {}
    for raw_ticker, stock_info in raw_metadata.items():
        ticker = normalize_ticker(raw_ticker)
        if ticker in metadata:
            raise ValueError(f"Duplicate normalized ticker in metadata: {ticker}")
        if not isinstance(stock_info, dict):
            raise ValueError(f"Metadata for {ticker} must be a JSON object.")
        metadata[ticker] = stock_info
    return metadata


def _validate_metadata_level(value: str, field: str, ticker: str) -> str:
    if value != value.strip() or value in {"", ".", ".."}:
        raise ValueError(f"Unsafe metadata field {field!r} for {ticker}: {value!r}")
    if "/" in value or "\\" in value or "\x00" in value or Path(value).is_absolute():
        raise ValueError(f"Unsafe metadata field {field!r} for {ticker}: {value!r}")
    return value


def _assert_within_data_dir(path: Path) -> Path:
    data_root = DATA_DIR.resolve()
    try:
        path.resolve().relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Ticker data path escapes {DATA_DIR}: {path}") from exc
    return path


def _existing_casefold_path(path: Path) -> Path | None:
    """Resolve an existing path below data even when component case differs."""
    _assert_within_data_dir(path)
    try:
        relative_parts = path.relative_to(DATA_DIR).parts
    except ValueError as exc:
        raise ValueError(f"Ticker data path escapes {DATA_DIR}: {path}") from exc

    current = DATA_DIR
    for part in relative_parts:
        if not current.is_dir():
            return None

        matches = [child for child in current.iterdir() if child.name.casefold() == part.casefold()]
        if not matches:
            return None
        if len(matches) > 1:
            formatted = ", ".join(str(match) for match in sorted(matches))
            raise AmbiguousTickerDataError(
                f"Multiple case-insensitive path matches for {path}: {formatted}"
            )
        current = matches[0]

    if not current.is_file():
        return None
    return _assert_within_data_dir(current)


def canonical_filepath(ticker: str, metadata: dict | None = None) -> Path:
    """Return the required storage path, whether or not the CSV exists yet."""
    normalized = normalize_ticker(ticker)
    metadata = load_stock_metadata() if metadata is None else metadata
    stock_info = metadata.get(normalized)

    if stock_info is None:
        return _assert_within_data_dir(UNPLACED_DIR / f"{normalized}.csv")

    levels = []
    for field in ("sector", "industry", "subindustry"):
        value = stock_info.get(field)
        if value:
            if not isinstance(value, str):
                raise ValueError(f"Metadata field {field!r} for {normalized} must be a string.")
            levels.append(_validate_metadata_level(value, field, normalized))

    return _assert_within_data_dir(
        DATA_DIR.joinpath(*levels, "_stocks", f"{normalized}.csv")
    )


def _stock_folders() -> list[Path]:
    folders = [
        path
        for path in DATA_DIR.rglob("*")
        if path.is_dir() and path.name.casefold() == "_stocks"
    ]
    return sorted(_assert_within_data_dir(path) for path in folders)


def _hierarchical_matches(ticker: str) -> list[Path]:
    normalized = normalize_ticker(ticker)
    matches = []
    for stock_folder in _stock_folders():
        matches.extend(
            _assert_within_data_dir(path)
            for path in stock_folder.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".csv"
            and path.stem.casefold() == normalized.casefold()
        )
    return sorted(set(matches))


def _local_filepath(ticker: str, metadata: dict) -> Path:
    normalized = normalize_ticker(ticker)
    canonical_path = canonical_filepath(normalized, metadata)
    existing_canonical_path = _existing_casefold_path(canonical_path)

    if normalized in metadata:
        if existing_canonical_path is not None:
            return existing_canonical_path
        raise FileNotFoundError(
            f"Metadata exists for {normalized}, but its CSV is missing: {canonical_path}"
        )

    if existing_canonical_path is not None:
        return existing_canonical_path

    # Transitional compatibility for currently unclassified files that already
    # live below an _stocks directory. New unclassified downloads use UnPlaced.
    matches = _hierarchical_matches(normalized)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        formatted = ", ".join(str(path) for path in matches)
        raise AmbiguousTickerDataError(
            f"Multiple CSV files found for unclassified ticker {normalized}: {formatted}"
        )

    raise FileNotFoundError(
        f"Could not find {normalized}.csv at {canonical_path} or below an _stocks folder."
    )


def local_filepath(ticker: str) -> Path:
    """Resolve one existing canonical or compatible local CSV for a ticker."""
    return _local_filepath(ticker, load_stock_metadata())


def ticker_exists(ticker: str) -> bool:
    metadata = load_stock_metadata()
    try:
        _local_filepath(ticker, metadata)
    except FileNotFoundError:
        return False
    return True


def _add_file_ticker(candidates: set[str], path: Path) -> None:
    try:
        candidates.add(normalize_ticker(path.stem))
    except (TypeError, ValueError):
        # Unrelated or malformed CSV names are not valid ticker candidates.
        pass


def discover_ticker_files() -> dict[str, Path]:
    """Return one resolvable local CSV per ticker, excluding legacy root files."""
    metadata = load_stock_metadata()
    candidates = set(metadata)

    for directory in DATA_DIR.iterdir():
        if directory.is_dir() and directory.name.casefold() == "unplaced":
            _assert_within_data_dir(directory)
            for path in directory.iterdir():
                if path.is_file() and path.suffix.casefold() == ".csv":
                    _add_file_ticker(candidates, path)

    for stock_folder in _stock_folders():
        for path in stock_folder.iterdir():
            if path.is_file() and path.suffix.casefold() == ".csv":
                _add_file_ticker(candidates, path)

    resolved = {}
    for ticker in sorted(candidates):
        try:
            resolved[ticker] = _local_filepath(ticker, metadata)
        except (FileNotFoundError, AmbiguousTickerDataError):
            # Missing and ambiguous tickers are not advertised as resolvable.
            continue
    return resolved


def discover_tickers() -> list[str]:
    return list(discover_ticker_files())


def iter_local_data_files() -> list[Path]:
    """Return canonical local data files in deterministic ticker order."""
    return list(discover_ticker_files().values())


def ambiguous_unclassified_tickers() -> dict[str, list[Path]]:
    """Return ambiguous legacy locations that have no metadata or UnPlaced file."""
    metadata = load_stock_metadata()
    candidates = set()
    for stock_folder in _stock_folders():
        for path in stock_folder.iterdir():
            if path.is_file() and path.suffix.casefold() == ".csv":
                _add_file_ticker(candidates, path)

    ambiguous = {}
    for ticker in sorted(candidates - set(metadata)):
        unplaced_path = canonical_filepath(ticker, metadata)
        if _existing_casefold_path(unplaced_path) is not None:
            continue
        matches = _hierarchical_matches(ticker)
        if len(matches) > 1:
            ambiguous[ticker] = matches
    return ambiguous


def missing_metadata_tickers() -> list[str]:
    metadata = load_stock_metadata()
    return sorted(
        ticker
        for ticker in metadata
        if _existing_casefold_path(canonical_filepath(ticker, metadata)) is None
    )


def load_local_data(ticker: str) -> pd.DataFrame:
    """Load the resolved local ticker CSV and prepare a Date index."""
    path = local_filepath(ticker)
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    return df


def _resolve_data_dir(data_dir: str | None = None) -> Path:
    target_dir = Path(data_dir) if data_dir is not None else UNPLACED_DIR
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir
    return target_dir


def ensure_ticker_data(
    ticker: str,
    start: str = "1986-08-20",
    end: str | None = None,
    data_dir: str | None = None,
    auto_adjust: bool = False,
) -> Path:
    normalized = normalize_ticker(ticker)
    metadata = load_stock_metadata()
    expected_path = canonical_filepath(normalized, metadata)
    target_dir = expected_path.parent

    if data_dir is not None:
        requested_dir = _resolve_data_dir(data_dir)
        _assert_within_data_dir(requested_dir / f"{normalized}.csv")
        if requested_dir.resolve() != target_dir.resolve():
            raise ValueError(
                f"data_dir for {normalized} must be its canonical directory "
                f"{target_dir}, not {requested_dir}."
            )

    try:
        return _local_filepath(normalized, metadata)
    except FileNotFoundError:
        pass

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(
        download_ticker_data(
            normalized,
            start=start,
            end=end,
            data_dir=str(target_dir),
            auto_adjust=auto_adjust,
        )
    )

    if downloaded_path.resolve() != expected_path.resolve():
        raise RuntimeError(
            f"Downloader saved {normalized} to {downloaded_path}; expected {expected_path}."
        )
    if not downloaded_path.is_file():
        raise FileNotFoundError(
            f"Expected downloaded file not found for {normalized}: {downloaded_path}"
        )
    return downloaded_path
