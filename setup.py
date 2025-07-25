from setuptools import find_packages
from setuptools import setup

install_requires = [
    'numpy',
    'scipy',
    'gymnasium==0.29.1',
    'pfrl@git+https://github.com/prabhatnagarajan/pfrl@mujoco_experiments',
    'matplotlib',
    'typing_extensions==4.8.0',
    'minatar',
    ]

test_requires = [
    'pytest',
    'attrs<19.2.0',  # pytest does not run with attrs==19.2.0 (https://github.com/pytest-dev/pytest/issues/3280)  # NOQA
]

setup(
    name='reg-duel-q',
    version='1.0.0',
    description='Code release for An Analysis of Action-Value Temporal-Difference Methods That Learn State Values',
    keywords='dueling Q-learning, dqn, QV-learning',
    author='Brett Daley and Prabhat Nagarajan',
    author_email='nagarajan@ualberta.ca',
    packages=find_packages(),
    install_requires=install_requires,
    test_requires=test_requires
)
