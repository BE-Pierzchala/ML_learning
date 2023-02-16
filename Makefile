clean_code: isort black interrogate

black:
	black .

isort:
	isort .

interrogate:
	interrogate -vv -i -I