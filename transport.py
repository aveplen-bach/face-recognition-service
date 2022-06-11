import os

import grpc
import face_recognition
import numpy as np
import sys
import logging
import time
from logging import StreamHandler, Formatter


from protos.facerec import facerec_pb2, facerec_pb2_grpc
from protos.s3file import s3file_pb2, s3file_pb2_grpc


class Facerec(facerec_pb2_grpc.FaceRecognition):

    def __init__(self):
        channel = grpc.insecure_channel("localhost:30031")
        self.stub = s3file_pb2_grpc.S3GatewayStub(channel)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = StreamHandler(stream=sys.stdout)
        handler.setFormatter(Formatter(fmt="[%(asctime)s: %(levelname)s] %(message)s"))
        self.logger.addHandler(handler)

        self.logger.debug("debug information")

    def ExtractFFVectorV1(self, request, context):
        nowstr = str(time.time())
        try:
            self.logger.info(f"ExtractFFVectorV1 called with ObjectID: {request.id}")

            response = self.stub.GetImageObject(s3file_pb2.GetImageObjectRequest(id=request.id))
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

            self.logger.info(f"FFVectorDistance called with len(ffva): {len(ffvcaar)}, len(ffvb): {len(ffvcbar)}")

            return facerec_pb2.FFVectorDistanceResponse(
                distance=np.linalg.norm(ffvcaar - ffvcbar, axis=0).item()
            )
        except Exception as e:
            print(e)
