# Makefile for automating Python project tasks
.PHONY: clean
.PHONY: Init


Init: 
	mkdir -p output	
	mkdir -p output/tables	
	mkdir -p output/figures	

# Variables
PYTHON = python3
MAIN = main.py
VENV = .venv
PIP = $(VENV)/bin/pip
MAKE = make

# Virtual environment setup
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

# Install dependencies if requirements.txt is present
install: $(VENV)/bin/activate
	if [ -f requirements.txt ]; then \
		$(PIP) install --upgrade pip; \
		$(PIP) install -r requirements.txt; \
	else \
		echo "Warning: requirements.txt not found. Skipping dependency installation."; \
	fi

# Generate all outputs: figures and tables
model: install
	$(PYTHON) $(MAIN)

# Generate only figures
figures: install
	$(PYTHON) $(MAIN) --figures

# Generate only tables
tables: install
	$(PYTHON) $(MAIN) --tables

# Clean generated figures and tables
clean:
	rm -rf output/figures/*
	rm -rf output/tables/*

# Remove the virtual environment
clean-venv:
	rm -rf $(VENV)

# Full clean: remove venv and all generated files
full-clean: clean clean-venv

# Help command
help:
	@echo "Available commands:"
	@echo "  make install      - Set up the virtual environment and install dependencies"
	@echo "  make model        - Generate all figures and tables including both feature visulization and modelling"
	@echo "  make figures      - Generate only figures of feature visulization"
	@echo "  make tables       - Generate only tables of exploratory data analysis"
	@echo "  make clean        - Remove generated figures and tables"
	@echo "  make clean-venv   - Remove the virtual environment"
	@echo "  make full-clean   - Remove generated files and the virtual environment"
	@echo "  make help         - Show this help message"