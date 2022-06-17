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
        self.logger.info(
            f"ExtractFFVectorV1 called with ObjectID: {request.id}")
        try:
            filename = str(request.id)

            self.logger.debug("getting image object from s3")
            response = self.stub.GetImageObject(
                s3file_pb2.GetImageObjectRequest(id=request.id))

            self.logger.debug("extracting contents from s3 response")
            img_blob = response.contents

            self.logger.debug("saving contents as file")
            with open(filename, "wb") as f:
                self.logger.debug(f"creating file {filename}")
                f.write(img_blob)

            self.logger.debug("loading file with face_recogntion")
            img = face_recognition.load_image_file(filename)

            # handling errors like a pro
            self.logger.debug("extracting encoding")
            encodings = face_recognition.face_encodings(img)
            if len(encodings) != 1:

                if len(encodings) < 1:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "there are no faces on the provided photo")
                    return facerec_pb2.ExtractFFVectorV1Response()

                if len(encodings) > 1:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "there are more then one faces on the provided photo")
                    return facerec_pb2.ExtractFFVectorV1Response()

            self.logger.debug("responding to caller")
            return facerec_pb2.ExtractFFVectorV1Response(ffvc=encodings[0].tolist())

        except Exception as e:
            self.logger.error(str(e))

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return facerec_pb2.ExtractFFVectorV1Response()

        finally:
            self.logger.debug("removing file from os")
            if os.path.exists(filename):
                self.logger.debug(f"removing file {filename}")
                os.remove(filename)

    def FFVectorDistance(self, request, context):
        self.logger.info("FFVectorDistance called")

        try:
            self.logger.debug("conferting vector a to np array")
            ffvcaar = np.asarray(request.ffvca)

            self.logger.debug("conferting vector b to np array")
            ffvcbar = np.asarray(request.ffvcb)

            self.logger.debug("responding to caller")
            return facerec_pb2.FFVectorDistanceResponse(
                distance=np.linalg.norm(ffvcaar - ffvcbar, axis=0).item()
            )
        except Exception as e:
            self.logger.debug("logging exception")
            print(e)
            self.logger.error(e)


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
    # t2 = threading.Thread(target=g.run)
    t1.start()
    # t2.start()
    g.run()
