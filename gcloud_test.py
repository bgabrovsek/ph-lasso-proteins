from google.cloud import storage

uniprot_id = 'A0A000'

"""Downloads a blob from the bucket."""                                 
# The ID of your GCS bucket                                             
bucket_name = "public-datasets-deepmind-alphafold-v4"                   
                                                                        
# The ID of your GCS object                                             
source_blob_name = f"AF-{uniprot_id}-F1-model_v4.cif"                   
file_path = source_blob_name
                                                                        
# The path to which the file should be downloaded                       
storage_client = storage.Client()                                       
bucket = storage_client.bucket(bucket_name)                             
                                                                        
# Construct a client side representation of a blob.                     
# Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
# any content from Google Cloud Storage. As we don't need additional data,
# using `Bucket.blob` is preferred here.                                
blob = bucket.blob(source_blob_name)                                    

blob.download_to_filename(file_path)                                    
                                                                        
print("Downloaded storage object {} from bucket {} to local file {}.".format(
      source_blob_name, bucket_name, file_path))       
