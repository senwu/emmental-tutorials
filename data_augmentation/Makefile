dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

check:
	isort -rc -c bin/image
	isort -rc -c eda/
	black bin/ --check
	black eda/ --check
	flake8 bin/
	flake8 eda/

clean:
	pip uninstall -y eda
	rm -rf eda.egg-info pip-wheel-metadata

format:
	isort -rc bin/image
	isort -rc eda/
	black bin/
	black eda/
	flake8 bin/
	flake8 eda/

.PHONY: dev check clean