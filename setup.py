from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vigipy", 
    version="0.2.1",  
    author="David Beery",  
    author_email="shakesbeery@gmail.com",  
    description="A Python library for disproportionality analyses",  
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shakesbeery/vigipy", 
    project_urls={
        "Bug Tracker": "https://github.com/Shakesbeery/vigipy/issues",  
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"}, 
    packages=find_packages(where="src"),
    python_requires=">=3.7",  
    install_requires=[
        "pandas==2.2.2",
        "numpy<2",
        "scipy==1.13.1",
        "scikit-learn==1.5.1",
        "sympy==1.12",
        "statsmodels==0.14.2",
    ],
    include_package_data=True,
)
