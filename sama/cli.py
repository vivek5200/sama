"""Sama CLI — command-line interface for the calibrated decision engine."""
import argparse
import json
import sys


def cmd_decide(args):
    """Run a single decision from the command line."""
    from sama import DecisionEngine

    engine = DecisionEngine.from_pretrained(args.model)

    questions = {
        "q": {
            "type": "choice",
            "instructions": args.instructions,
            "options": args.options,
        }
    }

    result = engine.decide(state=args.state, questions=questions)
    print(result.model_dump_json(indent=2))


def cmd_serve(args):
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install sama[server]",
              file=sys.stderr)
        sys.exit(1)

    # Set model path as env var so server.py can read it
    import os
    os.environ["SAMA_MODEL"] = args.model

    uvicorn.run(
        "sama.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="sama",
        description="Sama — calibrated decision engine",
    )
    sub = parser.add_subparsers(dest="command")

    # --- decide ---
    p_decide = sub.add_parser("decide", help="Run a single decision")
    p_decide.add_argument("--model", default="Vivek1225/sama-qwen-1.5b-v0.2",
                          help="HuggingFace repo ID or local path")
    p_decide.add_argument("--state", required=True,
                          help="Context / state text")
    p_decide.add_argument("--instructions", required=True,
                          help="Question instructions")
    p_decide.add_argument("--options", nargs="+", required=True,
                          help="Choice options")

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start the FastAPI server")
    p_serve.add_argument("--model", default="Vivek1225/sama-qwen-1.5b-v0.2",
                         help="HuggingFace repo ID or local path")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "decide":
        cmd_decide(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
