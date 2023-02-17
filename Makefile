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

clean_code: isort black interrogate

black:
	black .

isort:
	isort .

interrogate:
	interrogate -vv -i -I

export_env:
	conda env export --no-builds | grep -v "prefix" > enviroment.yml
create_env:
	conda env create -f enviromnt.yml