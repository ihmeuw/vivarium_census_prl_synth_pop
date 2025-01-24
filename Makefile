# Check if we're running in Jenkins
ifdef JENKINS_URL
	# Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
	# For local dev, search in parent directory
	MAKE_INCLUDES := ../vivarium_build_utils/resources/makefiles
endif

PACKAGE_NAME = vivarium_census_prl_synth_pop

# Include the makefiles
include $(MAKE_INCLUDES)/base.mk
include $(MAKE_INCLUDES)/test.mk

.PHONY: install
install: ## Install setuptools, package, and build utilities
	pip install --upgrade pip setuptools 
	pip install -e .[DEV]
