from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def download_folder_files(folder_id):
    gauth = GoogleAuth()
    gauth.CommandLineAuth()
    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    for file in file_list:
        print('title: %s, id: %s' % (file['title'], file['id']))
        file.GetContentFile(file['title'])


download_folder_files("1AAhTbNroSFsygHBXa88emCU7f50MxI8t")
