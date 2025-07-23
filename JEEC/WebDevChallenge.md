# Character Manager – Cheat-Sheet

## 1. Database & Models
* SQLAlchemy `engine`, `db_session`, `Base`
* `Character` model  
  `id`, `created_at`, `name`, `age`, `birthday`, `gender (enum)`, `is_alive`
* `initDB.py` → `Base.metadata.create_all(bind=engine)`

## 2. Finders & Handlers
| Camada  | Responsabilidade | Métodos-chave |
|---------|------------------|---------------|
| **Finder**   | Leitura        | `get_by_id`, `get_by_name`, `filter` |
| **Handler**  | Escrita        | `create`, `update`, `delete` (commit/rollback) |

## 3. API Routes
| Método | Rota                                   | Ação                          |
|--------|----------------------------------------|-------------------------------|
| GET    | `/characters`                          | Lista + filtros               |
| GET    | `/characters/&lt;id&gt;`               | Detalhe de 1 personagem       |
| POST   | `/characters`                          | Criar personagem              |
| PATCH  | `/characters/&lt;id&gt;`               | Atualizar campos              |
| DELETE | `/characters/&lt;id&gt;`               | Eliminar personagem           |
| GET    | `/characters/search?name=&lt;str&gt;` | Procurar por nome exato       |

## 4. Front-end (Vue 3 + Axios)
* `src/services/api.js` centraliza o Axios (`baseURL: localhost:5000`).
* **Componentes**  
  `CharacterForm.vue` → POST  
  `CharacterList.vue` → GET / PATCH / DELETE  
  `CharacterSearch.vue` → GET search
* Layout Grid: formulário (roxo) à esquerda, lista (azul) à direita.

## 5. Quick cURL Commands
```bash
# Criar
curl -X POST -H "Content-Type: application/json" \
  -d '{"name":"Jyn Erso","age":21,"birthday":"2003-05-25","gender":"female","is_alive":false}' \
  http://127.0.0.1:5000/characters

# Listar
curl http://127.0.0.1:5000/characters

# Obter
curl http://127.0.0.1:5000/characters/<id>

# Atualizar
curl -X PATCH -H "Content-Type: application/json" \
  -d '{"is_alive":true}' \
  http://127.0.0.1:5000/characters/<id>

# Eliminar
curl -X DELETE http://127.0.0.1:5000/characters/<id>

# Procurar
curl "http://127.0.0.1:5000/characters/search?name=Jyn%20Erso"
