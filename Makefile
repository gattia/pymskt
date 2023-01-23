.PHONY: build install 

requirements:
	python -m pip install -r requirements.txt

requirements-mamba:
	mamba install --file requirements.txt

requirements-conda:
	conda install --file requirements.txt

build:
	python -m build -o wheelhouse

install:
	pip install .

install-dev: 
	pip install --editable .

coverage: 
	coverage run -m pytest
	# Make the html version of the coverage results. 
	coverage html 

dev:
	python -m pip install --upgrade -r requirements-dev.txt

dev-mamba:
	mamba install --file requirements-dev.txt

dev-conda:
	conda install --file requirements-dev.txt
	
docs:
	pdoc --output-dir docs/ --html --force pymskt
	mv docs/pymskt/* docs/
	rm -rf docs/pymskt

test:
	set -e
	pytest

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .

autoformat:
	set -e
	isort .
	black --config pyproject.toml .