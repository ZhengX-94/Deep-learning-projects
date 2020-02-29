import requests

url = 'http://127.0.0.1:5000/predict'

r = requests.post(url, json={'comment': "This film was just brilliant casting location scenery story direction \
                        everyone's really suited the part they played and you could just imagine being there robert"})
print(r.json())