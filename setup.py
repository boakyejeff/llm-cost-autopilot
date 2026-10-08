from setuptools import find_packages, setup

setup(
    name="cost-autopilot",
    version="0.1.0",
    description="Track LLM API usage and cost, enforce budgets, route to the cheapest capable model",
    author="Jeffrey B. Appiagyei",
    packages=find_packages(),
    python_requires=">=3.9",
)
