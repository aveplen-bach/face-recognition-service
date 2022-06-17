import os
import grpc
import sys
import logging
import time
import threading
from logging import StreamHandler, Formatter
from http.server import BaseHTTPRequestHandler, HTTPServer
from concurrent import futures

import yaml
import numpy as np
import face_recognition
from yaml.loader import SafeLoader

from protos.facerec import facerec_pb2, facerec_pb2_grpc
from protos.s3file import s3file_pb2, s3file_pb2_grpc


class LifecheckServer():
    def __init__(self, cfg):
        self.cfg = cfg

    class LifecheckHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            print("feeling healthy")
            self.send_response(200, "OK")
            self.end_headers()

    def run(self):
        addr = self.cfg["lifecheck"]["addr"]
        port = int(addr[len(addr)-4:])

        server_address = ('', port)
        httpd = HTTPServer(server_address, self.LifecheckHandler)
        print(f"lifecheck server is running on {port}")
        httpd.serve_forever()


class Facerec(facerec_pb2_grpc.FaceRecognition):

    def __init__(self, cfg):
        channel = grpc.insecure_channel(cfg["s3-client"]["addr"])
        self.stub = s3file_pb2_grpc.S3GatewayStub(channel)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            Formatter(fmt="[%(asctime)s: %(levelname)s] %(message)s"))
        self.logger.addHandler(handler)

        self.logger.debug("debug information")

    def ExtractFFVectorV1(self, request, context):
        nowstr = str(time.time())
        try:
            self.logger.info(
                f"ExtractFFVectorV1 called with ObjectID: {request.id}")

            response = self.stub.GetImageObject(
                s3file_pb2.GetImageObjectRequest(id=request.id))
            img_blob = response.contents

            with open(nowstr, "wb") as f:
                self.logger.info(f"creating file {nowstr}")
                f.write(img_blob)
                img = face_recognition.load_image_file(nowstr)
                img_face_encoding = face_recognition.face_encodings(img)[0]
                return facerec_pb2.ExtractFFVectorV1Response(ffvc=img_face_encoding)
        except Exception as e:
            print(e)

        finally:
            if os.path.exists(nowstr):
                self.logger.info(f"removing file {nowstr}")
                os.remove(nowstr)

    def FFVectorDistance(self, request, context):
        try:
            ffvcaar = np.asarray(request.ffvca)
            ffvcbar = np.asarray(request.ffvcb)

            self.logger.info(
                f"FFVectorDistance called with len(ffva): {len(ffvcaar)}, len(ffvb): {len(ffvcbar)}")

            return facerec_pb2.FFVectorDistanceResponse(
                distance=np.linalg.norm(ffvcaar - ffvcbar, axis=0).item()
            )
        except Exception as e:
            print(e)


class GrpcServer():
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        addr = os.environ.get("GRPC_LISTEN_ADDR", self.cfg["server"]["addr"])
        port = int(addr[len(addr)-5:])

        print(f"grpc server is running on {port}")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        facerec_pb2_grpc.add_FaceRecognitionServicer_to_server(
            Facerec(self.cfg), server)
        server.add_insecure_port(f"[::]:{port}")
        server.start()
        server.wait_for_termination()


def read_config(filename):
    with open(filename) as file:
        cfg = yaml.load(file, Loader=SafeLoader)
        return cfg


if __name__ == '__main__':
    cfg = read_config("facerec-service.yaml")

    l = LifecheckServer(cfg)
    g = GrpcServer(cfg)

    t1 = threading.Thread(target=l.run)
    t2 = threading.Thread(target=g.run)
    t1.start()
    t2.start()
