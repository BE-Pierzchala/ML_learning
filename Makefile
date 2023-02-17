make test_LR:
	@echo "Preparing data"
	python Experiments/LinearRegression/prep_data.py
	@echo "Scikit"
	python Experiments/LinearRegression/LR_Scikit.py
	@echo "Mine"
	python Experiments/LinearRegression/my_LR.py

clean_code: isort black interrogate

black:
	black .

isort:
	isort .

interrogate:
	interrogate -vv -i -I