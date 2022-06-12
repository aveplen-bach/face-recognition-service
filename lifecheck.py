from http.server import BaseHTTPRequestHandler, HTTPServer


class LifecheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)


def run(cfg, server_class=HTTPServer, handler_class=LifecheckHandler):
    addr = cfg["lifecheck"]["addr"]
    port = int(addr[len(addr)-4:])

    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"lifecheck server is running on {port}")
    httpd.serve_forever()
