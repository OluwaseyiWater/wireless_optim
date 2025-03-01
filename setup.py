from setuptools import setup, find_packages

setup(
    name="purejaxrl",
    version="0.1.0",
    description="Really Fast End-to-End JAX RL Implementations",
    author="luchris429",
    url="https://github.com/luchris429/purejaxrl",
    packages=find_packages(),
    install_requires=[
        "jax>=0.5.0",
        "jaxlib>=0.5.0",
        "optax",
        "dm-haiku",
        "distrax",
    ],
    python_requires=">=3.8",
)