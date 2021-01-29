import requests
import json

# URL for the web service, should be similar to:
# 'http://17fb482f-c5d8-4df7-b011-9b33e860c111.southcentralus.azurecontainer.io/score'
scoring_uri = 'http://41cbb5ad-1b92-4bf3-aead-e134d7a13903.southcentralus.azurecontainer.io/score'
#key= ''

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
          },
          {
            "Pregnancies": 7,
            "Glucose": 105,
            "BloodPressure": 0,
            "SkinThickness": 0,
            "Insulin": 0,
            "BMI": 0,
            "DiabetesPedigreeFunction": 0.305,
            "Age": 24
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
#headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


