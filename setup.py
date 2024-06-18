from setuptools import setup, find_packages

setup(
    name = 'r3kit', 
    version = '0.0.0', 
    license = 'MIT', 
    description = 'Research kits for real robots', 
    author = "Junbo Wang", 
    author_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    maintainer = "Junbo Wang", 
    maintainer_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    url = "https://github.com/dadadadawjb/r3kit", 
    packages = find_packages(), 
    include_package_data = True, 
    install_requires = [
        'numpy', 
        'transformations', 
    ], 
    zip_safe = False
)
