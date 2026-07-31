# Zepto Data & AI Platform

> End-to-end AI & Data Engineering Capstone Project demonstrating Data Engineering, Machine Learning, and Generative AI through a fictional Zepto business scenario.

---

# Project Overview

This repository showcases the complete lifecycle of an AI-powered data platform, from data collection and analytics to building an intelligent support assistant.

The project is divided into three independent modules that together demonstrate production-oriented software engineering and AI development practices.

---

# Project Modules

## Module 1 — Data Pipeline ✅

Build an end-to-end ETL pipeline that:

- Scrapes product data from BooksToScrape
- Cleans and transforms the dataset
- Converts prices from GBP to INR
- Loads data into a normalized SQLite database
- Executes SQL queries
- Demonstrates SQL and Pandas integration

**Status:** Completed

---

## Module 2 — Analytics 🚧

Perform data analysis and predictive modeling by:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Development
- Model Evaluation
- Business Insights

**Status:** In Progress

---

## Module 3 — AI Support Assistant 🚧

Develop a GenAI-powered customer support assistant using:

- Retrieval-Augmented Generation (RAG)
- Vector Database
- LangGraph Workflow
- FastAPI
- Hugging Face Deployment

**Status:** In Progress

---

# Repository Structure

```text
zepto-data-ai-platform/

├── analytics/
├── data_pipeline/
├── support_assistant/
│
├── CHANGELOG.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Technologies

## Data Engineering

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite

## Machine Learning

- Scikit-learn
- NumPy
- Matplotlib

## Generative AI

- LangChain
- LangGraph
- ChromaDB
- Hugging Face
- FastAPI

## Development Tools

- Git
- GitHub
- VS Code

---

# Getting Started

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run a module by navigating to its directory and following the module-specific README.

Example:

```bash
cd data_pipeline
```

---

# Current Progress

| Module | Status |
|---------|--------|
| Data Pipeline | ✅ Completed |
| Analytics | 🚧 In Progress |
| AI Support Assistant | 🚧 In Progress |

---

# Future Improvements

- Containerize applications using Docker
- Add CI/CD pipeline
- Deploy services to the cloud
- Improve monitoring and logging
- Expand AI assistant capabilities

---

# Author

**Tejesh Bandla**