#!/usr/bin/env python3
"""Teste rápido dos endpoints da API."""

import sys
import requests
import time
import subprocess
import os
import signal

# Start the server
proc = subprocess.Popen(['python', 'api_memoria.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(3)

try:
    # Test health endpoint
    r = requests.get('http://localhost:8000/health')
    print(f'Health: {r.status_code} - {r.json()}')
    
    # Test recall endpoint
    r = requests.get('http://localhost:8000/memoria/recall', params={'consulta': 'RAG'})
    print(f'Recall: {r.status_code} - {len(r.json()["resultados"])} resultados')
    
    # Test stats endpoint
    r = requests.get('http://localhost:8000/memoria/stats')
    print(f'Stats: {r.status_code} - {r.json()}')
    
    print('All endpoints working!')
finally:
    # Kill the server process
    os.kill(proc.pid, signal.SIGTERM)