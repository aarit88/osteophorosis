import urllib.request
import json

url = "http://127.0.0.1:8000/openapi.json"
data = json.loads(urllib.request.urlopen(url).read())
schemas = data.get("components", {}).get("schemas", {})

target_schemas = [
    "BiomechanicsGeometryRequest",
    "BiomechanicsMirelsRequest",
    "ExplainabilityDemoRequest",
    "RadiomicsDemoRequest",
    "UncertaintyDemoRequest",
    "SyntheticWorkbenchRunRequest",
    "BlockedInferenceResponse"
]

for s_name in target_schemas:
    s = schemas.get(s_name, {})
    print(f"\n--- {s_name} ---")
    props = s.get("properties", {})
    required = s.get("required", [])
    for p_name, p_info in props.items():
        is_req = "(REQUIRED)" if p_name in required else f"(default={p_info.get('default')})"
        print(f"  {p_name:<25} : {p_info.get('type', p_info.get('$ref')):<15} {is_req}")

