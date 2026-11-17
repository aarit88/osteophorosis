import httpx
import json

base_url = "http://127.0.0.1:8000"
client = httpx.Client(base_url=base_url, timeout=30.0)

def inspect(name, resp):
    print(f"\n=== {name} (Status: {resp.status_code}) ===")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2)[:500])
    except Exception as e:
        print("Raw text:", resp.text[:300])

inspect("tracks", client.get("/api/v1/research/tracks"))
inspect("evidence", client.get("/api/v1/evidence/registry"))
inspect("readiness", client.get("/api/v1/architecture/readiness"))
inspect("radiomics", client.post("/api/v1/radiomics/demo", json={"phantom_type": "checkerboard"}))
inspect("mirels", client.post("/api/v1/biomechanics/mirels", json={"site": 2, "pain": 2, "radiographic_nature": 3, "lesion_size": 2}))
inspect("explainability", client.post("/api/v1/explainability/demo", json={"target_class": 1}))
inspect("four-class", client.post("/api/v1/diagnostic/four-class", json={}))

