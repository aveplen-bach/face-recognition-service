import threading
import lifecheck
import server
import yaml
from yaml.loader import SafeLoader


def read_config(filename):
    with open(filename) as file:
        cfg = yaml.load(file, Loader=SafeLoader)
        return cfg


if __name__ == '__main__':
    cfg = read_config("facerec-service.yaml")
    t1 = threading.Thread(target=server.run, args=(cfg,))
    t2 = threading.Thread(target=lifecheck.run, args=(cfg,))
    t1.run()
    t2.run()
