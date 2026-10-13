import tempfile
import json
from pathlib import Path
from mnemo.permissoes import Permissoes
from mnemo.indexador import Indexador
from mnemo.ferramentas import FerramentasVault
import sqlite3

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
        # Check raw database
        conn = f.indexador.con
        cursor = conn.execute("SELECT * FROM nota_tags")
        print("nota_tags table:")
        for row in cursor:
            print(f"  {row}")
        
        cursor = conn.execute("SELECT * FROM notas")
        print("\nnotas table:")
        for row in cursor:
            print(f"  {row}")
        
        # Check the query used in list_tags
        print("\n--- list_tags query ---")
        cursor = conn.execute("""
            SELECT tag, COUNT(DISTINCT nota_id) as count
            FROM nota_tags
            GROUP BY tag
            ORDER BY count DESC, tag
        """)
        for row in cursor:
            print(f"  {row}")