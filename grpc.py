from .protos.facerec import facerec_pb2, facerec_pb2_grpc

class Facerec(facerec_pb2_grpc.FaceRecognition):
    def ExtractFFVectorV1(self, request, context):
        print(f"=================\n{request}")
        return facerec_pb2.ExtractFFVectorV1Response(ffvc=[1.1, 2.2, 3.3, 4.4])
