from google.cloud import storage

BILLING_PROJECT = "phlasso"
BUCKET = "public-datasets-deepmind-alphafold-v4"
OBJECT = "AF-A0A000-F1-model_v4.cif"

client = storage.Client(project=BILLING_PROJECT)

bucket = client.bucket(BUCKET, user_project=BILLING_PROJECT)
blob = bucket.blob(OBJECT)

data = blob.download_as_bytes()
print("Downloaded bytes:", len(data))