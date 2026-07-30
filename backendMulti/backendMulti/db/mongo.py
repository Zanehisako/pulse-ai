from pymongo import MongoClient
from django.conf import settings
import gridfs

_client = None
_db = None
_fs = None

def get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(settings.MONGO_URI)
        _db = _client[settings.MONGO_DB_NAME]
    return _db

def get_fs():
    global _fs
    if _fs is None:
        _fs = gridfs.GridFS(get_db())
    return _fs