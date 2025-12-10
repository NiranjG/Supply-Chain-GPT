"""
FastAPI application for SupplyChainGPT
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .dependencies import get_services
from ..config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting SupplyChainGPT API...")
    settings.ensure_directories()

    # Initialize services (lazy loading will happen on first use)
    services = get_services()
    logger.info("Services container initialized")

    yield

    # Shutdown
    logger.info("Shutting down SupplyChainGPT API...")


# Create FastAPI app
app = FastAPI(
    title="SupplyChainGPT API",
    description="""
    RAG-Powered Generative AI Co-Pilot for Smart Inventory Planning

    ## Features
    - **Ask Questions**: Get AI-powered answers from your supply chain documents
    - **Document Management**: Upload and manage business documents
    - **Demand Forecasting**: Get SKU-level demand predictions
    - **Safety Stock**: Calculate optimal safety stock levels
    - **Hybrid Search**: Find relevant information using semantic + keyword search

    ## Authentication
    Currently using simple user context. Production should implement proper auth.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SupplyChainGPT",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
