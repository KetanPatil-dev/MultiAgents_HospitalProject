#!/bin/sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path-to-level>"
  exit 1
fi

LEVEL="$1"
PYTHONPATH=searchclient_python java -jar server.jar -l "$LEVEL" -c "python3 -m searchclient.client --bypass-debugger" -t 180 -s 300 -g