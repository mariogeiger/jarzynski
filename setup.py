from setuptools import find_packages, setup

setup(
    name='jarzynski',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'jax',
    ],
    include_package_data=True,
    python_requires='>=3.7',
)
