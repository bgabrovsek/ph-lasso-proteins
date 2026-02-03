from google.cloud import storage
from google.auth import default

creds, project = default()
print("Project:", project)

client = storage.Client(credentials=creds, project=project)
print("Client created")

