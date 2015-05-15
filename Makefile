notebook: spark_log_stats/sparklogstats
	export PYTHONPATH="$$PYTHONPATH:spark_log_stats"; \
	ipython3 notebook

spark_log_stats/sparklogstats:
	git submodule init
	git submodule update
