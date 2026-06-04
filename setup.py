from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='dyslexia',
      description="Dyslexia detection package",
      packages=find_packages(),
      install_requires=requirements)
