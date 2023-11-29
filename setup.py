import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CrystalNet",                            
    version="1.0",                            
    author="Yi Yao, Lin Li",
    author_email="yyao94@asu.edu",                        
    description="Python package appling GNN into crystal materials",
    long_description=long_description,          
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),        
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                         
    python_requires='>=3.10',                  
    py_modules=["CrystalNet"],                             
    install_requires=[]                       
)