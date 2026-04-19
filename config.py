"""
config.py

Central configuration for QRAG-Tax.
Reads secrets from environment; uses sane defaults for the rest.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (alongside this file) before reading any os.getenv calls.
load_dotenv(Path(__file__).parent / ".env")

# Retrieval
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "30"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))
MAX_TOKENS_CONTEXT = int(os.getenv("MAX_TOKENS_CONTEXT", "3000"))

# Vector store
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "tax_strategies")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# Quantum (optional — D-Wave Leap)
DWAVE_API_TOKEN = os.getenv("DWAVE_API_TOKEN")

# IBM Quantum (Qiskit Runtime)
IBM_QUANTUM_API_KEY = os.getenv("IBM_QUANTUM_API_KEY")
IBM_QUANTUM_CRN = os.getenv("IBM_QUANTUM_CRN")
IBM_QUANTUM_CHANNEL = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_cloud")
