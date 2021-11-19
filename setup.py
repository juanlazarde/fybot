"""Install packages

Command::

        pip install .
"""
import os
import setuptools

README = os.path.join(os.path.dirname(__file__), 'README.md')
REQUIREMENT = os.path.join(os.path.dirname(__file__), 'requirements.txt')


setuptools.setup(
    include_package_data=True,
    name='fybot',
    version='0.0.1',
    packages=setuptools.find_packages(),
    package_dir={'': 'fybot'},
    package_data=setuptools.find_namespace_packages(),
    namespace_packages=['fybot'],
    url='https://github.com/juanlazarde/fybot',
    license='Apache License',
    author='Juan Lazarde & Alejandro Ramirez',
    author_email='',
    description='Financial Bot in Python',
    long_description=open(README).read() + "\n\n",
    keywords="finance invest bot options stock nasdaq",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        x.replace("==", ">=")
        for x in open(REQUIREMENT).read().split('\n')
        if x != ""
    ],
)
