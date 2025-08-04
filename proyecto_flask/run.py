# run.py
from app import create_app

app = create_app()

# DEBUG: listar rutas
print("\nRUTAS REGISTRADAS:")
for rule in app.url_map.iter_rules():
    print(f"{rule.methods} → {rule.rule} (endpoint={rule.endpoint})")
print()

if __name__ == '__main__':
    app.run(debug=True)