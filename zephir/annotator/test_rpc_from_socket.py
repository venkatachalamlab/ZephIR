import requests

r = requests.post(
    'http://localhost:5001/socket',
    json={"method": "jump_to_frame", "arg": "200"},
    headers={"Content-Type": "application/json"},
)

print(r.text)
