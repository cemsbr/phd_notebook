# Include "lib" and "spark_log_stats" folders in PYTHONPATH
# Usage: ". env.sh" or "source env.sh"

# this script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$DIR:$DIR/spark_log_stats:$DIR/notebooks:$PYTHONPATH"
. venv/bin/activate
