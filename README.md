# Data Toolkit

A collection of general and specific reusable and scalable functions for data scientists and data engineers to accelerate their process in the fields:

1. Data Science:
    - NLP Binary and Multiclass Classifications.
    - Time-series forecasting.
2. Data Engineering:
    - Solutions connected via streamlit (python frontend framework).
    - Interaction with infrastructure resources on Google and Microsoft Azure Clouds.

[![PyPI version](https://img.shields.io/pypi/v/data-toolkit?style=for-the-badge)](https://pypi.org/project/data-toolkit/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/amilton23/data_toolkit/main.yml?branch=main&style=for-the-badge)](https://github.com/amilton23/data_toolkit/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## Package Structure

This toolkit is organized into two main subpackages for clarity and ease of use:

```
.
├── models/    # A dedicated module for implementations, wrappers, or classes of Machine Learning and Deep Learning models.
│   ├── bert.py: Classes, methods and helper functions for using BERT models, likely for NLP tasks.
│   ├── decisiontree.py: Functions of decision tree-based models, as well as their optimizations.
│   ├── lstm.py: Classes, methods and helper functions for using LSTM models, common in time-series.
│   └── genai.py: Classes, methods and helper functions of GenAI Gemini and GPT models.
├── tutorials/     # Folder containing usual examples for the functions. 
│   └── ...
├── utils/     # The core of the toolkit, containing helper functions for the entire data project lifecycle.
│   ├── integration/ # A subpackage for interacting with cloud platforms and data APIs.
│   │   └── google.py: Helpers for Google Cloud Platform (e.g., BigQuery, Cloud Storage).
│   │   └── microsoft.py: Helpers for Microsoft Azure (e.g., Blob Storage, SQL Database).
│   ├── general.py: General-purpose functions that don't fit into other categories (e.g., file I/O, logging).
│   ├── preprocessing.py: Functions for data cleaning, transformation, encoding, and feature engineering.
│   ├── statistics.py: Functions for statistical analysis (e.g., hypothesis testing, correlation).
│   └── viz.py: Functions for creating standardized and reusable data visualizations.
└── ...
```

## Installation

You can install `data-toolkit` from different ways, depending on your necessity.

**Option 01: Via PyPI (Recommended)**
```bash
pip install data-toolkit
```
