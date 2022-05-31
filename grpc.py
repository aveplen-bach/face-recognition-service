import os

import grpc
import face_recognition
import numpy as np

from .protos.facerec import facerec_pb2, facerec_pb2_grpc
from .protos.s3file import s3file_pb2, s3file_pb2_grpc



class Facerec(facerec_pb2_grpc.FaceRecognition):

    def __init__(self):
        channel = grpc.insecure_channel("localhost:9090")
        self.stub = s3file_pb2_grpc.S3GatewayStub(channel)

    def ExtractFFVectorV1(self, request, context):
        response = self.stub.GetImageObject(s3file_pb2.GetImageObjectRequest(id=request.id))
        img_blob = response.contents

        if os.path.exists("tmp"):
            os.remove("tmp")

        with open("tmp", "wb") as f:
            f.write(img_blob)
            img = face_recognition.load_image_file("tmp")
            img_face_encoding = face_recognition.face_encodings(img)[0]
            return facerec_pb2.ExtractFFVectorV1Response(ffvc=img_face_encoding)
