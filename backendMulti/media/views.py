from urllib import request

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import HttpResponse
from .services import upload_file, get_file, delete_file
import mimetypes

class UploadMediaView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    

    def post(self, request):
        file = request.FILES.get('file')

        if not file:
            return Response({"error": "No file provided"}, status=400)

        if not file.name.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.mov')):
            return Response({"error": "Invalid file type"}, status=400)

        if file.size > 50 * 1024 * 1024:
            return Response({"error": "File too large"}, status=400)

        file_id = upload_file(file)

        return Response({
            "message": "Uploaded successfully",
            "file_id": file_id
        } ,status=201)


class GetMediaView(APIView):
    def get(self, request, file_id):
        try:
            file = get_file(file_id)

            content_type , _ = mimetypes.guess_type(file.filename)

            return HttpResponse(
                file.read(),
                content_type=content_type or 'application/octet-stream',
                headers={
                    "Content-Disposition": f'inline; filename="{file.filename}"'
                }
            )
        except:
            return Response({"error": "Not found"}, status=404)


class DeleteMediaView(APIView):
    def delete(self, request, file_id):
        try:
            delete_file(file_id)
            return Response({"message": "Deleted"})
        except:
            return Response({"error": "Not found"}, status=404)