# Vercel API route for serverless Flask app
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Flask app - this creates the app at module load time
from web_app import app as flask_app

def handler(request, context):
    """Vercel Python handler"""
    try:
        # Vercel may pass a request object with an 'environ' attribute, or it may pass the WSGI environ dict directly.
        environ = getattr(request, "environ", request)

        response_body = []
        status = None
        headers = None

        def start_response(s, h, exc_info=None):
            nonlocal status, headers
            status = s
            headers = h

        result = flask_app.wsgi_app(environ, start_response)

        try:
            for part in result:
                response_body.append(part)
        finally:
            if hasattr(result, "close"):
                result.close()

        body_bytes = b"".join(response_body)
        status_code = int(status.split()[0]) if status else 200

        return {
            "statusCode": status_code,
            "headers": dict(headers) if headers else {},
            "body": body_bytes.decode("utf-8", errors="replace"),
        }
    except Exception as e:
        # Return error response
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": f"{{\"error\": \"{str(e)}\"}}",
        }
