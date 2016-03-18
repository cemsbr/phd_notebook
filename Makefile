notebook: spark_log_stats/sparklogstats
	. ./env.sh && jupyter notebook --no-browser

spark_log_stats/sparklogstats:
	git submodule init
	git submodule update
