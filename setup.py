"""
Setup script for SupplyChainGPT
"""

from setuptools import setup, find_packages

setup(
    name="supplychaingpt",
    version="1.0.0",
    description="RAG-Powered Generative AI Co-Pilot for Smart Inventory Planning",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "streamlit>=1.31.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "sentence-transformers>=2.3.0",
        "chromadb>=0.4.22",
        "openai>=1.12.0",
        "langchain>=0.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "prophet>=1.1.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.4.0",
        "pdfminer.six>=20231228",
        "python-docx>=1.1.0",
        "openpyxl>=3.1.0",
        "plotly>=5.18.0",
        "rank-bm25>=0.2.2",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "httpx>=0.26.0",
        ]
    },
)
