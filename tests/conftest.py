"""
Pytest configuration and fixtures for SupplyChainGPT tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    # Inventory Policy Document

    ## Overview
    This document outlines the inventory management policy for warehouse WH-001.

    ## Safety Stock Calculation
    Safety stock for SKU-12345 is calculated as follows:
    - Average daily demand: 100 units
    - Lead time: 7 days
    - Service level: 95%
    - Safety stock: 200 units

    ## Reorder Point
    The reorder point for SKU-12345 is 900 units.

    ## Supplier Information
    Primary supplier: SUP-001 (Acme Manufacturing)
    Lead time: 14 days
    """


@pytest.fixture
def sample_historical_data():
    """Sample historical demand data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    base_demand = 100
    demand = base_demand + np.random.randn(90) * 20
    demand = np.maximum(demand, 0)

    return pd.DataFrame({
        'ds': dates,
        'y': demand
    })


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_pdf_content():
    """Sample text that would be extracted from a PDF"""
    return """
    Supply Chain Report Q4 2024

    Executive Summary
    This quarterly report provides an overview of supply chain performance
    for the fourth quarter of 2024.

    Key Metrics:
    - On-time delivery rate: 97.5%
    - Inventory turnover: 8.2
    - Fill rate: 98.3%

    Warehouse Performance:
    WH-001: 95% capacity utilization
    WH-002: 82% capacity utilization
    WH-003: 78% capacity utilization

    SKU Analysis:
    Top performing SKUs: SKU-10001, SKU-10002, SKU-20001
    Slow movers: SKU-30003, SKU-30004
    """


@pytest.fixture
def sample_chunks():
    """Sample chunks for retrieval testing"""
    return [
        {
            "chunk_id": "chunk-001",
            "chunk_text": "Safety stock for SKU-12345 is 200 units based on 95% service level.",
            "metadata": {
                "doc_id": "doc-001",
                "doc_title": "Inventory Policy",
                "doc_type": "policy",
                "source_uri": "/docs/inventory_policy.pdf",
                "sku_ids": "SKU-12345",
                "warehouse_ids": "WH-001"
            },
            "similarity": 0.92
        },
        {
            "chunk_id": "chunk-002",
            "chunk_text": "Reorder point calculation: ROP = (Daily Demand × Lead Time) + Safety Stock",
            "metadata": {
                "doc_id": "doc-001",
                "doc_title": "Inventory Policy",
                "doc_type": "policy",
                "source_uri": "/docs/inventory_policy.pdf"
            },
            "similarity": 0.85
        },
        {
            "chunk_id": "chunk-003",
            "chunk_text": "SUP-001 (Acme Manufacturing) has a lead time of 14 days with 98% on-time delivery.",
            "metadata": {
                "doc_id": "doc-002",
                "doc_title": "Supplier Contracts",
                "doc_type": "contract",
                "source_uri": "/docs/supplier_contracts.pdf",
                "supplier_ids": "SUP-001"
            },
            "similarity": 0.78
        }
    ]


@pytest.fixture
def mock_user_context():
    """Mock user context for testing"""
    from src.models.schemas import UserContext
    return UserContext(
        user_id="test_user",
        tenant_id="default",
        roles=["planner", "viewer"]
    )


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for inventory data"""
    return """sku_id,warehouse_id,quantity,date
SKU-12345,WH-001,500,2024-01-15
SKU-12345,WH-002,300,2024-01-15
SKU-10001,WH-001,1000,2024-01-15
SKU-10001,WH-003,750,2024-01-15
SKU-20001,WH-001,200,2024-01-15
"""
