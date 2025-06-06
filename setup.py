from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(filepath:str) -> List[str]:
    """
    this function will return the list of requirements
    """
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
    
    # we have written the "-e ." in requirements.txt so that it can connect setup.py file but here we have to remove that -e . 
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='mlproject',
    version= '0.0.1',
    author='Smit',
    author_email='smitladani54578@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)