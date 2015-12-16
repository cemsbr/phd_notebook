#!/bin/bash

if [ $# -lt 1 ]; then
    echo 'Required argument: spark history file.'
    exit 1
fi

echo '['
cat $1 | while read line; do
    json=$(echo "$line" | json_pp)
    if [ $? -eq 0 ]; then
        echo "$json,"
    fi
done
echo ']'

>&2 echo 'remove the last comma'
