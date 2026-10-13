import tempfile
import json
from pathlib import Path
from mnemo.permissoes import Permissoes
from mnemo.indexador import Indexador
from mnemo.ferramentas import FerramentasVault

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / 'notas' / 'projetos').mkdir(parents=True)
    (tmp / 'notas' / 'estudo').mkdir(parents=True)
    (tmp / '.vault').mkdir()
    (tmp / '.vault' / 'config.json').write_text(
        json.dumps({'versao': 1, 'todas_as_pastas': False, 'pastas_permitidas': ['projetos', 'estudo']}),
        encoding='utf-8',
    )
    
    nota1 = tmp / 'notas' / 'projetos' / 'reuniao.md'
    nota1.write_text('---\ntags: [trabalho, reuniao, importante]\n---\n\n# Reunião\nConteúdo.\n', encoding='utf-8')
    
    nota2 = tmp / 'notas' / 'projetos' / 'tarefas.md'
    nota2.write_text('---\ntags: [trabalho, tarefa]\n---\n\n# Tarefas\nLista.\n', encoding='utf-8')
    
    with FerramentasVault(str(tmp)) as f:
        r = f.executar('list_tags', {'prefixo': 'tra'})
        print('Resultado:', r)
        if r['ok']:
            for t in r['tags']:
                print(f'  {t["tag"]} ({t["count"]})')