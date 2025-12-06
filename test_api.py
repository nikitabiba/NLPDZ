import requests

res = requests.post(
    "http://localhost:8000/api/ask/",
    json={"question": "Что такое Российская Федерация?"}
)

print(res.json())

# python manage.py runserver
# python test_api.py