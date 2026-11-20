#!/bin/bash
cd /opt/nodej-s-app
source /opt/.venv/bin/activate
gunicorn -w 4 -b 0.0.0.0:10000 main:app
