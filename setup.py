from setuptools import setup, find_packages

import mlflow_fastapi

setup(
   name=mlflow_fastapi.__name__,
   version=mlflow_fastapi.__version__,
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.com',
   packages=find_packages(),
   install_requires=[]
)
