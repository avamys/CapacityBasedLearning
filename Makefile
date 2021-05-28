.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = CapacityBasedLearning
PYTHON_INTERPRETER = python
UCI_DATA_URL_OBESITY = "https://archive.ics.uci.edu/ml/machine-learning-databases/00544/ObesityDataSet_raw_and_data_sinthetic%20(2).zip"
UCI_DATA_URL_MICE = https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls
UCI_DATA_URL_DRUGS = https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data
UCI_DATA_URL_CARS = https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data
UCI_DATA_URL_ADULT = https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
UCI_DATA_URL_ANNEALING = https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	@echo ">>> Downloading data from UCI."
	curl -o data/raw/obesity.zip $(UCI_DATA_URL_OBESITY)
	unzip data/raw/obesity.zip -d data/raw/
	find data/raw/ -type f ! -iname "*.csv" -delete
	mv data/raw/* data/raw/obesity.csv
	curl -o data/raw/mice.xls $(UCI_DATA_URL_MICE)
	curl -o data/raw/adult.data $(UCI_DATA_URL_ADULT)
	curl -o data/raw/anneal.data $(UCI_DATA_URL_ANNEALING)
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/adult.data data/processed/
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/obesity.csv data/processed/
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/mice.xls data/processed/
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/anneal.data data/processed/

## Data preprocessing
preprocessing: 
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/adult.data data/processed/
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/obesity.csv data/processed/
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/mice.xls data/processed/
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/anneal.data data/processed/
	
## Train Model
train:
	@echo ">>> Training model"
	$(PYTHON_INTERPRETER) src/models/train_model.py data/processed/adult_features.csv data/processed/adult_target.csv experiments/experiment_test.yaml

## Run tests
run_tests:
	@echo ">>> Running unit tests"
	$(PYTHON_INTERPRETER) -m unittest discover tests

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
