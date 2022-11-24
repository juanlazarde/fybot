#!/bin/bash
# set -e

# psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
# 	CREATE USER docker;
# 	CREATE DATABASE docker;
# 	GRANT ALL PRIVILEGES ON DATABASE docker TO docker;
# EOSQL

# echo "running fybot entry point"
# python3 /usr/src/fybot

# echo "running create tables entrypoint"
# python3 /usr/src/fybot/config/create_tables.py

#!/bin/sh

# other initialization steps

# /entrypoint-parent.sh "$@" &

# potentially wait for parent to be running by polling

# run something new in the foreground, that may depend on parent processes
# exec /usr/bin/mongodb ..