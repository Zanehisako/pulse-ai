from backendMulti.db.mongo import get_fs
from bson import ObjectId

def upload_file(file):
    fs = get_fs()

    file_id = fs.put(
        file,
        filename=file.name,
        content_type=file.content_type
    )

    return str(file_id)

def get_file(file_id):
    fs = get_fs()
    return fs.get(ObjectId(file_id))

def delete_file(file_id):
    fs = get_fs()
    fs.delete(ObjectId(file_id))