import urllib.request
import json

url = "http://127.0.0.1:8000/openapi.json"
data = json.loads(urllib.request.urlopen(url).read())

print("OpenAPI Title:", data["info"]["title"])
print("Version:", data["info"]["version"])
print("Total paths:", len(data["paths"]))
print("=" * 80)

for path, methods in sorted(data["paths"].items()):
    for method, details in methods.items():
        summary = details.get("summary", "")
        req_body = details.get("requestBody", {})
        schema_ref = ""
        if req_body:
            content = req_body.get("content", {})
            app_json = content.get("application/json", {})
            schema = app_json.get("schema", {})
            schema_ref = schema.get("$ref", schema.get("title", ""))
        print(f"{method.upper():<6} {path:<45} {summary:<30} {schema_ref}")

print("=" * 80)
print("COMPONENTS / SCHEMAS:")
for name in sorted(data.get("components", {}).get("schemas", {}).keys()):
    print(" -", name)

