import os

def carregar_env(caminho='.env'):
    try:
        with open(caminho, encoding='utf-8') as f:
            for linha in f:
                linha = linha.strip()
                if not linha or linha.startswith('#') or '=' not in linha:
                    continue
                chave, valor = linha.split('=', 1)
                os.environ.setdefault(chave.strip(), valor.strip().strip("'\""))
                print('Set', chave.strip(), '=', valor.strip().strip("'\""))
    except FileNotFoundError:
        print('File not found')

carregar_env()
print('NVIDIA_API_KEY:', os.environ.get('NVIDIA_API_KEY', 'NOT SET'))