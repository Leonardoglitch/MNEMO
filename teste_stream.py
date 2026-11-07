# teste_stream.py
from mnemo.modelo_nvidia import ClienteNVIDIA

c = ClienteNVIDIA()
for chunk in c.conversar([{"role": "user", "content": "Diz olá em 3 línguas"}], stream=True):
    content = chunk["delta"].get("content")
    if content:
        print(content, end="", flush=True)
print()