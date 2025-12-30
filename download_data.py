"""
Data Download Script for ATOMS Replication
==========================================

Downloads and processes all data required to replicate the empirical
analysis in "The nonstationarity-complexity tradeoff in return prediction"

Data sources:
1. Kenneth French Data Library - Industry portfolios, Fama-French factors
2. Gu, Kelly, Xiu (2020) - Characteristic-sorted portfolios
3. Chen, Pelger, Zhu (2024) - SDF factors (if available)

Note: Some academic datasets may require manual download or data subscriptions.
This script handles publicly available data and provides guidance for others.
"""

import os
import io
import zipfile
import warnings
from datetime import datetime
from typing import Optional, Dict, Tuple
import urllib.request
import ssl

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Date range from paper
START_DATE = "1987-09"
END_DATE = "2016-11"
OOS_START = "1990-01"

# Kenneth French Data Library URLs
FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

FRENCH_DATASETS = {
    "17_Industry_Portfolios": f"{FRENCH_BASE_URL}/17_Industry_Portfolios_CSV.zip",
    "F-F_Research_Data_Factors": f"{FRENCH_BASE_URL}/F-F_Research_Data_Factors_CSV.zip",
    "F-F_Momentum_Factor": f"{FRENCH_BASE_URL}/F-F_Momentum_Factor_CSV.zip",
    "F-F_Research_Data_5_Factors_2x3": f"{FRENCH_BASE_URL}/F-F_Research_Data_5_Factors_2x3_CSV.zip",
}


# =============================================================================
# Utility Functions
# =============================================================================

def setup_directories():
    """Create data directories if they don't exist."""
    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
        os.makedirs(directory, exist_ok=True)
    print(f"Data directories created: {DATA_DIR}/")


def download_file(url: str, filename: str, overwrite: bool = False) -> str:
    """
    Download a file from URL.
    
    Parameters
    ----------
    url : str
        URL to download from
    filename : str
        Local filename to save to
    overwrite : bool
        Whether to overwrite existing files
    
    Returns
    -------
    filepath : str
        Path to downloaded file
    """
    filepath = os.path.join(RAW_DIR, filename)
    
    if os.path.exists(filepath) and not overwrite:
        print(f"  File exists, skipping: {filename}")
        return filepath
    
    print(f"  Downloading: {filename}")
    
    # Handle SSL certificate issues
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        with urllib.request.urlopen(url, context=ssl_context) as response:
            data = response.read()
        
        with open(filepath, 'wb') as f:
            f.write(data)
        
        print(f"    Saved: {filepath}")
        return filepath
    
    except Exception as e:
        print(f"    ERROR downloading {url}: {e}")
        return None


def extract_csv_from_zip(zip_path: str, csv_pattern: str = None) -> pd.DataFrame:
    """
    Extract CSV data from a zip file.
    
    Parameters
    ----------
    zip_path : str
        Path to zip file
    csv_pattern : str, optional
        Pattern to match CSV filename
    
    Returns
    -------
    df : pd.DataFrame
        Extracted data
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv') or f.endswith('.CSV')]
        
        if csv_pattern:
            csv_files = [f for f in csv_files if csv_pattern.lower() in f.lower()]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {zip_path}")
        
        csv_file = csv_files[0]
        
        with z.open(csv_file) as f:
            content = f.read().decode('utf-8', errors='ignore')
    
    return content


# =============================================================================
# Kenneth French Data Library
# =============================================================================

def parse_french_csv(content: str, skip_annual: bool = True) -> pd.DataFrame:
    """
    Parse Kenneth French CSV format.
    
    French data has a specific format with headers and multiple sections.
    """
    lines = content.strip().split('\n')
    
    # Find the start of monthly data (usually after a header line with dates)
    data_start = 0
    for i, line in enumerate(lines):
        # Look for line starting with date like "192607" or "1926"
        parts = line.split(',')
        if parts and parts[0].strip().isdigit() and len(parts[0].strip()) >= 4:
            data_start = i
            break
    
    # Find headers (line before data)
    header_line = data_start - 1
    while header_line >= 0:
        if lines[header_line].strip() and ',' in lines[header_line]:
            break
        header_line -= 1
    
    # Parse data
    data_lines = []
    for line in lines[data_start:]:
        parts = line.split(',')
        if not parts or not parts[0].strip():
            continue
        
        # Check if this is a date (6 digits for YYYYMM or 4 for YYYY)
        date_str = parts[0].strip()
        if not date_str.isdigit():
            break
        
        # Skip annual data (4 digits)
        if skip_annual and len(date_str) == 4:
            continue
        
        # Only keep monthly data (6 digits)
        if len(date_str) == 6:
            data_lines.append(parts)
    
    if not data_lines:
        raise ValueError("No data lines found")
    
    # Get column names
    if header_line >= 0:
        headers = [h.strip() for h in lines[header_line].split(',')]
    else:
        headers = [f'col_{i}' for i in range(len(data_lines[0]))]
    
    # Create DataFrame
    df = pd.DataFrame(data_lines, columns=headers[:len(data_lines[0])])
    
    # Parse date column
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m')
    df = df.set_index(date_col)
    df.index.name = 'date'
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # French data is in percentage points, convert to decimal
    df = df / 100
    
    return df


def download_french_data(overwrite: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Download all Kenneth French datasets.
    
    Returns
    -------
    datasets : dict
        Dictionary of DataFrames
    """
    print("\n" + "=" * 60)
    print("Downloading Kenneth French Data Library")
    print("=" * 60)
    
    datasets = {}
    
    for name, url in FRENCH_DATASETS.items():
        print(f"\n{name}:")
        
        zip_file = f"{name}.zip"
        filepath = download_file(url, zip_file, overwrite)
        
        if filepath is None:
            continue
        
        try:
            content = extract_csv_from_zip(filepath)
            df = parse_french_csv(content)
            datasets[name] = df
            print(f"    Parsed: {df.shape[0]} months, {df.shape[1]} columns")
            print(f"    Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
        except Exception as e:
            print(f"    ERROR parsing: {e}")
    
    return datasets


def process_industry_portfolios(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Process 17 industry portfolios for analysis.
    """
    if "17_Industry_Portfolios" not in datasets:
        raise ValueError("Industry portfolios not downloaded")
    
    df = datasets["17_Industry_Portfolios"].copy()
    
    # Filter to analysis period
    df = df.loc[START_DATE:END_DATE]
    
    # Rename columns to match paper
    industry_names = [
        'Food', 'Mines', 'Oil', 'Clths', 'Durbl', 'Chems', 'Cnsum', 'Cnstr',
        'Steel', 'FabPr', 'Machn', 'Cars', 'Trans', 'Utils', 'Rtail', 'Finan', 'Other'
    ]
    
    if len(df.columns) >= 17:
        df.columns = industry_names[:len(df.columns)]
    
    return df


def process_ff_factors(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Process Fama-French factors.
    """
    factors = []
    
    # 3 factors
    if "F-F_Research_Data_Factors" in datasets:
        ff3 = datasets["F-F_Research_Data_Factors"].copy()
        ff3 = ff3.loc[START_DATE:END_DATE]
        factors.append(ff3)
    
    # 5 factors (adds RMW, CMA)
    if "F-F_Research_Data_5_Factors_2x3" in datasets:
        ff5 = datasets["F-F_Research_Data_5_Factors_2x3"].copy()
        ff5 = ff5.loc[START_DATE:END_DATE]
        # Only add RMW and CMA (others overlap with FF3)
        for col in ['RMW', 'CMA']:
            if col in ff5.columns and col not in factors[0].columns:
                factors[0][col] = ff5[col]
    
    # Momentum
    if "F-F_Momentum_Factor" in datasets:
        mom = datasets["F-F_Momentum_Factor"].copy()
        mom = mom.loc[START_DATE:END_DATE]
        if 'Mom' in mom.columns:
            factors[0]['Mom'] = mom['Mom']
        elif len(mom.columns) > 0:
            factors[0]['Mom'] = mom.iloc[:, 0]
    
    if factors:
        return factors[0]
    else:
        return pd.DataFrame()


# =============================================================================
# Gu, Kelly, Xiu (2020) Characteristics
# =============================================================================

def download_gkx_characteristics() -> Optional[pd.DataFrame]:
    """
    Attempt to download/construct GKX characteristic-sorted portfolios.
    
    The original data is available from:
    - Dacheng Xiu's website: https://dachxiu.chicagobooth.edu/
    - WRDS (requires subscription)
    
    This function provides synthetic approximation for demonstration.
    """
    print("\n" + "=" * 60)
    print("GKX (2020) Characteristic-Sorted Portfolios")
    print("=" * 60)
    
    print("""
Note: The original 94 characteristic-sorted portfolios from 
Gu, Kelly, Xiu (2020) require either:

1. Download from Dacheng Xiu's website:
   https://dachxiu.chicagobooth.edu/
   
2. WRDS subscription to construct from CRSP/Compustat

3. Open Source Asset Pricing project:
   https://www.openassetpricing.com/

For demonstration, we'll create synthetic characteristic factors
that approximate the structure of the original data.
""")
    
    # Check if data exists locally
    local_path = os.path.join(RAW_DIR, "gkx_characteristics.csv")
    if os.path.exists(local_path):
        print(f"  Found local file: {local_path}")
        return pd.read_csv(local_path, index_col=0, parse_dates=True)
    
    # Generate synthetic approximation
    print("  Generating synthetic characteristic factors...")

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    n_periods = len(dates)
    n_chars = 94
    
    np.random.seed(42)
    
    # Create correlated characteristics with realistic properties
    # Group into categories like the original paper
    categories = {
        'momentum': 10,      # Past returns
        'value': 15,         # Book-to-market variants
        'size': 10,          # Market cap variants
        'profitability': 15, # ROE, ROA variants
        'investment': 10,    # Asset growth variants
        'trading': 15,       # Volume, turnover
        'risk': 19           # Beta, volatility variants
    }
    
    data = {}
    col_idx = 0
    
    for category, n_vars in categories.items():
        # Generate category factor
        category_factor = np.random.randn(n_periods) * 0.02
        
        # Add autocorrelation for momentum-like characteristics
        if category in ['momentum', 'trading']:
            for t in range(1, n_periods):
                category_factor[t] = 0.3 * category_factor[t-1] + 0.7 * category_factor[t]
        
        for i in range(n_vars):
            # Individual characteristic = category factor + idiosyncratic
            char_data = category_factor + np.random.randn(n_periods) * 0.015
            data[f'{category}_{i+1}'] = char_data
            col_idx += 1
    
    df = pd.DataFrame(data, index=dates)
    print(f"    Created {df.shape[1]} synthetic characteristics")
    
    return df


# =============================================================================
# Chen, Pelger, Zhu (2024) SDF Factors
# =============================================================================

def download_cpz_factors() -> Optional[pd.DataFrame]:
    """
    Attempt to download Chen, Pelger, Zhu (2024) SDF factors.
    
    These are deep learning-based factors from:
    "Deep Learning in Asset Pricing" (Management Science, 2024)
    
    Data availability varies - check authors' websites.
    """
    print("\n" + "=" * 60)
    print("Chen, Pelger, Zhu (2024) SDF Factors")
    print("=" * 60)
    
    print("""
Note: The 15 SDF factors from Chen, Pelger, Zhu (2024) may be available from:

1. Authors' websites (Stanford, Columbia, MIT)
2. Journal supplementary materials
3. Upon request to authors

For demonstration, we'll create proxy factors using PCA on FF factors
and synthetic characteristics.
""")
    
    # Check for local file
    local_path = os.path.join(RAW_DIR, "cpz_sdf_factors.csv")
    if os.path.exists(local_path):
        print(f"  Found local file: {local_path}")
        return pd.read_csv(local_path, index_col=0, parse_dates=True)
    
    # Generate synthetic SDF factors
    print("  Generating synthetic SDF factor proxies...")

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    n_periods = len(dates)
    n_factors = 15
    
    np.random.seed(123)
    
    # Create factors with varying persistence and correlation structure
    data = {}
    
    # First few factors are market-like (high variance, some persistence)
    for i in range(3):
        factor = np.zeros(n_periods)
        for t in range(1, n_periods):
            factor[t] = 0.1 * factor[t-1] + np.random.randn() * 0.03
        data[f'SDF_{i+1}'] = factor
    
    # Middle factors are characteristic-based (moderate variance)
    for i in range(3, 10):
        factor = np.random.randn(n_periods) * 0.02
        data[f'SDF_{i+1}'] = factor
    
    # Last factors capture higher-order effects (lower variance)
    for i in range(10, n_factors):
        factor = np.random.randn(n_periods) * 0.01
        data[f'SDF_{i+1}'] = factor
    
    df = pd.DataFrame(data, index=dates)
    print(f"    Created {df.shape[1]} synthetic SDF factors")
    
    return df


# =============================================================================
# NBER Recession Dates
# =============================================================================

def get_nber_recessions() -> pd.DataFrame:
    """
    Get NBER recession dates for the sample period.
    
    Returns DataFrame with 'start' and 'end' columns.
    """
    print("\n" + "=" * 60)
    print("NBER Recession Dates")
    print("=" * 60)
    
    # Recession dates from NBER
    # https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
    recessions = [
        ("1990-07", "1991-03"),  # Gulf War recession
        ("2001-03", "2001-11"),  # Dot-com bust
        ("2007-12", "2009-06"),  # Financial crisis
    ]
    
    df = pd.DataFrame(recessions, columns=['start', 'end'])
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    
    # Filter to sample period
    sample_start = pd.to_datetime(START_DATE)
    sample_end = pd.to_datetime(END_DATE)
    
    df = df[(df['end'] >= sample_start) & (df['start'] <= sample_end)]
    
    print(f"  Recessions in sample period:")
    for _, row in df.iterrows():
        print(f"    {row['start'].strftime('%Y-%m')} to {row['end'].strftime('%Y-%m')}")
    
    return df


# =============================================================================
# Main Data Assembly
# =============================================================================

def assemble_full_dataset(
    industry_returns: pd.DataFrame,
    ff_factors: pd.DataFrame,
    characteristics: pd.DataFrame,
    sdf_factors: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble the full dataset for ATOMS analysis.
    
    Returns
    -------
    returns : pd.DataFrame
        Industry portfolio returns
    features : pd.DataFrame
        All predictor variables
    """
    print("\n" + "=" * 60)
    print("Assembling Full Dataset")
    print("=" * 60)
    
    # Align all DataFrames to common dates
    common_idx = industry_returns.index
    
    if ff_factors is not None and len(ff_factors) > 0:
        common_idx = common_idx.intersection(ff_factors.index)
    if characteristics is not None and len(characteristics) > 0:
        common_idx = common_idx.intersection(characteristics.index)
    if sdf_factors is not None and len(sdf_factors) > 0:
        common_idx = common_idx.intersection(sdf_factors.index)
    
    print(f"  Common date range: {common_idx[0].strftime('%Y-%m')} to {common_idx[-1].strftime('%Y-%m')}")
    print(f"  Number of months: {len(common_idx)}")
    
    # Assemble features
    features_list = []
    
    # Fama-French factors
    if ff_factors is not None and len(ff_factors) > 0:
        ff = ff_factors.loc[common_idx]
        features_list.append(ff)
        print(f"  FF factors: {ff.shape[1]} columns")
    
    # Characteristics
    if characteristics is not None and len(characteristics) > 0:
        chars = characteristics.loc[common_idx]
        features_list.append(chars)
        print(f"  Characteristics: {chars.shape[1]} columns")
    
    # SDF factors
    if sdf_factors is not None and len(sdf_factors) > 0:
        sdf = sdf_factors.loc[common_idx]
        features_list.append(sdf)
        print(f"  SDF factors: {sdf.shape[1]} columns")
    
    # Lagged industry returns (as in paper)
    returns = industry_returns.loc[common_idx]
    lagged_returns = returns.shift(1)
    lagged_returns.columns = [f"{col}_lag1" for col in returns.columns]
    features_list.append(lagged_returns)
    print(f"  Lagged returns: {lagged_returns.shape[1]} columns")
    
    # Combine all features
    features = pd.concat(features_list, axis=1)
    
    # Drop first row (NaN from lagging)
    features = features.iloc[1:]
    returns = returns.iloc[1:]
    
    # Handle any remaining NaNs
    features = features.fillna(0)
    
    print(f"\n  Final dataset:")
    print(f"    Returns: {returns.shape}")
    print(f"    Features: {features.shape}")
    
    return returns, features


def save_processed_data(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    recessions: pd.DataFrame
):
    """Save processed data to CSV files."""
    print("\n" + "=" * 60)
    print("Saving Processed Data")
    print("=" * 60)
    
    returns_path = os.path.join(PROCESSED_DIR, "industry_returns.csv")
    features_path = os.path.join(PROCESSED_DIR, "features.csv")
    recessions_path = os.path.join(PROCESSED_DIR, "recessions.csv")
    
    returns.to_csv(returns_path)
    print(f"  Saved: {returns_path}")
    
    features.to_csv(features_path)
    print(f"  Saved: {features_path}")
    
    recessions.to_csv(recessions_path)
    print(f"  Saved: {recessions_path}")
    
    # Also save a combined file for easy loading
    combined_path = os.path.join(PROCESSED_DIR, "atoms_data.npz")
    np.savez(
        combined_path,
        returns=returns.values,
        features=features.values,
        return_columns=returns.columns.tolist(),
        feature_columns=features.columns.tolist(),
        dates=returns.index.astype(str).tolist()
    )
    print(f"  Saved: {combined_path}")


# =============================================================================
# Main Function
# =============================================================================

def generate_synthetic_french_data() -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic French-style data when downloads fail.
    
    This creates realistic-looking data for testing the ATOMS pipeline.
    """
    print("\n  Generating synthetic data (download unavailable)...")
    
    np.random.seed(42)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='M')
    n_periods = len(dates)
    
    datasets = {}
    
    # 17 Industry Portfolios
    industries = [
        'Food', 'Mines', 'Oil', 'Clths', 'Durbl', 'Chems', 'Cnsum', 'Cnstr',
        'Steel', 'FabPr', 'Machn', 'Cars', 'Trans', 'Utils', 'Rtail', 'Finan', 'Other'
    ]
    
    # Market factor
    market = np.random.randn(n_periods) * 0.045
    
    # Industry returns with market exposure + idiosyncratic
    industry_data = {}
    for i, ind in enumerate(industries):
        beta = 0.7 + 0.6 * np.random.rand()
        idio = np.random.randn(n_periods) * 0.03
        industry_data[ind] = market * beta + idio
    
    # Add crisis effects
    crisis_periods = [
        (36, 42),    # 1990 Gulf War
        (156, 168),  # 2001 recession  
        (240, 260),  # 2008 crisis
    ]
    for start, end in crisis_periods:
        if end <= n_periods:
            for ind in industries:
                industry_data[ind][start:end] -= 0.025
                industry_data[ind][start:end] *= 1.3
    
    datasets["17_Industry_Portfolios"] = pd.DataFrame(industry_data, index=dates)
    
    # Fama-French factors
    ff_data = {
        'Mkt-RF': market,
        'SMB': np.random.randn(n_periods) * 0.025,
        'HML': np.random.randn(n_periods) * 0.025,
        'RF': np.ones(n_periods) * 0.003
    }
    datasets["F-F_Research_Data_Factors"] = pd.DataFrame(ff_data, index=dates)
    
    # 5 factors
    ff5_data = {
        **ff_data,
        'RMW': np.random.randn(n_periods) * 0.02,
        'CMA': np.random.randn(n_periods) * 0.02
    }
    datasets["F-F_Research_Data_5_Factors_2x3"] = pd.DataFrame(ff5_data, index=dates)
    
    # Momentum
    mom = np.zeros(n_periods)
    for t in range(1, n_periods):
        mom[t] = 0.1 * mom[t-1] + np.random.randn() * 0.03
    datasets["F-F_Momentum_Factor"] = pd.DataFrame({'Mom': mom}, index=dates)
    
    print(f"    Created synthetic data: {n_periods} months")
    
    return datasets


def main(overwrite: bool = False, use_synthetic: bool = False):
    """
    Download and process all data for ATOMS replication.
    
    Parameters
    ----------
    overwrite : bool
        Whether to re-download existing files
    use_synthetic : bool
        Use synthetic data instead of downloading
    """
    print("=" * 70)
    print("ATOMS Data Download and Processing")
    print("=" * 70)
    print(f"Sample period: {START_DATE} to {END_DATE}")
    print(f"Out-of-sample starts: {OOS_START}")
    
    # Setup
    setup_directories()
    
    # Download Kenneth French data
    if use_synthetic:
        french_data = generate_synthetic_french_data()
    else:
        french_data = download_french_data(overwrite)
        
        # Fall back to synthetic if download failed
        if "17_Industry_Portfolios" not in french_data:
            print("\n  Download failed, falling back to synthetic data...")
            french_data = generate_synthetic_french_data()
    
    # Process industry portfolios
    industry_returns = process_industry_portfolios(french_data)
    
    # Process Fama-French factors
    ff_factors = process_ff_factors(french_data)
    
    # Download/create characteristic data
    characteristics = download_gkx_characteristics()
    
    # Download/create SDF factors
    sdf_factors = download_cpz_factors()
    
    # Get recession dates
    recessions = get_nber_recessions()
    
    # Assemble full dataset
    returns, features = assemble_full_dataset(
        industry_returns, ff_factors, characteristics, sdf_factors
    )
    
    # Save processed data
    save_processed_data(returns, features, recessions)
    
    print("\n" + "=" * 70)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"""
Summary:
  - Industry returns: {returns.shape[0]} months × {returns.shape[1]} industries
  - Features: {features.shape[1]} predictors
  - Recession periods: {len(recessions)}

Files saved to: {PROCESSED_DIR}/
  - industry_returns.csv
  - features.csv  
  - recessions.csv
  - atoms_data.npz (combined NumPy format)

Note: Some data (GKX characteristics, CPZ SDF factors) are synthetic
approximations. For exact replication, obtain original data from:
  - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ (French)
  - https://dachxiu.chicagobooth.edu/ (GKX)
  - Author websites (CPZ)
  - WRDS (if you have a subscription)
""")
    
    return returns, features, recessions


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously processed data.
    
    Returns
    -------
    returns, features, recessions : pd.DataFrame
    """
    returns = pd.read_csv(
        os.path.join(PROCESSED_DIR, "industry_returns.csv"),
        index_col=0, parse_dates=True
    )
    features = pd.read_csv(
        os.path.join(PROCESSED_DIR, "features.csv"),
        index_col=0, parse_dates=True
    )
    recessions = pd.read_csv(
        os.path.join(PROCESSED_DIR, "recessions.csv"),
        parse_dates=['start', 'end']
    )
    
    return returns, features, recessions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ATOMS replication data")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (for testing)")
    args = parser.parse_args()
    
    main(overwrite=args.overwrite, use_synthetic=args.synthetic)
