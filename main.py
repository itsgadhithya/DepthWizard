"""
DepthWizard API Server Runner
Usage:
    python main.py --host 127.0.0.1 --port 8000 --reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="DepthWizard V2 API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Starting DepthWizard V2 API Server on http://{args.host}:{args.port}")
    print(f"  Interactive Swagger API Documentation: http://{args.host}:{args.port}/docs")
    print(f"  Interactive ReDoc Documentation:       http://{args.host}:{args.port}/redoc")
    print("=" * 60)

    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
