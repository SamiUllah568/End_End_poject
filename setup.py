from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str) -> List[str]:
    
    # This function will returns list of the requirements.txt
    requirements=[]
    with open(file_path) as file:
        requirements =file.readlines()
        requirements = [req.replace('\n',"")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="END_TO-END_PROJECT",
    version='0.0.1',
    author ='Sami Ullah',
    author_email='sk2579784@gmai.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
)