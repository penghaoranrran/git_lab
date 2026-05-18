from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    port = 8000
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd server on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
        sys.exit(0)

if __name__ == "__main__":
    run()
