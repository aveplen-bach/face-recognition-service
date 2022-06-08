from concurrent import futures
import grpc

from protos.facerec import facerec_pb2_grpc
from transport import Facerec

class Server:

    @staticmethod
    def run():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        facerec_pb2_grpc.add_FaceRecognitionServicer_to_server(Facerec(), server)
        server.add_insecure_port('[::]:30032')
        server.start()

        print("server started")

        server.wait_for_termination()
