lin_reg:
	@echo "Preparing data"
	python Experiments/data_prep.py LinearRegression insurance.csv charges
	@echo "Scikit"
	python Experiments/LinearRegression/lin_reg_scikit.py
	@echo "Mine"
	python Experiments/LinearRegression/lin_reg_mine.py

log_reg:
	@echo "Preparing data"
	python Experiments/data_prep.py LogisticRegression adult.csv income
	@echo "Scikit"
	python Experiments/LogisticRegression/log_reg_scikit.py
	@echo "Mine"
	python Experiments/LogisticRegression/log_reg_mine.py

network:
	@echo "Preparing data"
	python Experiments/data_prep.py LogisticRegression adult.csv income
	@echo "Mine"
	python Experiments/NeuralNetwork/NN_mine.py

test:
	pytest Tests

clean_code: isort black interrogate lint

lint:
	flakehell lint .
black:
	black .

isort:
	isort .

interrogate:
	interrogate -vv -i -I

export_env:
	poetry export --without-hashes -f requirements.txt --output requirements.txt

install:
	pip install -r requirements.txt
