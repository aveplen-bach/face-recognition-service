import grpc
import face_recognition
import numpy as np

from .protos.facerec import facerec_pb2, facerec_pb2_grpc
from .protos.s3file import s3file_pb2, s3file_pb2_grpc



class Facerec(facerec_pb2_grpc.FaceRecognition):

    def __init__(self):
        channel = grpc.insecure_channel('localhost:8083')
        self.stub = s3file_pb2_grpc.S3GatewayStub(channel)

    def ExtractFFVectorV1(self, request, context):
        print(f"=================\n{request}")

        response = self.stub.Reply(s3file_pb2.GetImageObjectRequest(id=request.id))
        imgBlob = response.contents

        img = np.array(imgBlob)
        img_face_encoding = face_recognition.face_encodings(img)[0]
        print(img_face_encoding.shape)
        print(img_face_encoding)

        return facerec_pb2.ExtractFFVectorV1Response(ffvc=[1.1, 2.2, 3.3, 4.4])
