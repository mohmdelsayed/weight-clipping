from setuptools import setup, find_packages

setup(
    name="weight-clipping",
    version="1.0.0",
    description="weight-clipping",
    url="https://github.com/mohmdelsayed",
    author="Mohamed Elsayed",
    author_email="mohamedelsayed@ualberta.ca",
    packages=find_packages(exclude=["tests*"]),
)
