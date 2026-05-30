#!/bin/bash
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy
cd /Users/wcici/xiaozhi-server/main/xiaozhi-server
source venv/bin/activate
python app.py
