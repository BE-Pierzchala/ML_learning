# Learning Machine Learning

This is a repository for my learning experiments with ML,
it contains implementations of some algorithms.

Currently, I have only implemented linear and logistic regressions.

Recreate this environment:
1. You must have conda installed
2. Run ``make create_env`` - this will create a virtual environment (venv) for this project
3. Run ```conda activate ML``` - this will activate this venv ensuring full functionality.


You can see available commands in the Makefile, currently there are:
- ```make lin_reg``` Runs Scikit's and mine implementations of linear regression and provides outputs for comparison
- ```make log_reg``` Runs Scikit's and mine implementations of logistic regression and provides outputs for comparison
  - Note that my algorithm is slow
 