#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:16:33 2020

@author: pasq
"""
from __future__ import print_function
import os
from googleapiclient import discovery
from httplib2 import Http
from oauth2client import file, client, tools
from apiclient.http import MediaFileUpload,MediaIoBaseDownload

SCOPES = 'https://www.googleapis.com/auth/drive'
store = file.Storage('storage.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('/Users/pasq/Documents/ML/git/credentials.json', SCOPES)
    creds = tools.run_flow(flow, store)
DRIVE = discovery.build('drive', 'v3', http=creds.authorize(Http()))


#Create a new file in the specified folder, root otherwise
def uploadFile(local_file_name, upload_name, folder=None):
    file_metadata = {
        "name": upload_name,
        "mimeType": "*/*"
    }
    if folder is not None:
        file_metadata["parents"] = [folder]
        
    media = MediaFileUpload(local_file_name,
                            mimetype='*/*',
                            resumable=True)
    file = DRIVE.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print ('File ID: ' + file.get('id'))


#List folders
folders = DRIVE.files().list(q="mimeType='application/vnd.google-apps.folder'").execute().get("files")
for f in folders:
    print(f["name"] + " " + f["id"])



#List all files
files = DRIVE.files().list().execute().get('files', [])
for f in files:
    print(f['name'], f['mimeType'])


file_path="/Users/pasq/Documents/ML/git/generative_models/celeba/img_align_celeba"
celeba_list = os.listdir(file_path)


for i in celeba_list[500:]:
    uploadFile(file_path+"/"+i, i)
    print(i)

    


