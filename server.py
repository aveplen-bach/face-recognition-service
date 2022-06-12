from concurrent import futures
import grpc

from protos.facerec import facerec_pb2_grpc
from transport import Facerec


def run(cfg):
    addr = cfg["server"]["addr"]
    port = int(addr[len(addr)-5:])

    print(f"grpc server is running on {port}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    facerec_pb2_grpc.add_FaceRecognitionServicer_to_server(
        Facerec(cfg), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
