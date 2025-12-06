import requests
import argparse
import sys
import os
from config import url

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("template", help="Path to template")
    parser.add_argument("--thresh", type=float, default=0.4)
    args = parser.parse_args()

    abs_path = os.path.abspath(args.template)
    
    # Save image on server if debug is true
    payload = {"path": abs_path, "thresh": args.thresh, "debug": True}

    try:
        resp = requests.post(url, json=payload, timeout=5)
        
        if resp.status_code != 200:
            print(f"Server error: {resp.status_code}\n{resp.text}")
            sys.exit(2)

        data = resp.json()
        
        if "error" in data:
            print(f"Error: {data['error']}")
            sys.exit(2)

        score = data['score']
        
        if data['found']:
            print(f"Found template: {os.path.basename(abs_path)}")
            print(f"   Similarity:  {score:.2%} (Próg: {args.thresh})")
            print(f"   Position:  X={data['x']}, Y={data['y']}")
            print(f"   Time taken:     {data['duration']:.4f}s")
            sys.exit(0)
        else:
            print(f"Failed to find template: {os.path.basename(abs_path)}!")
            print(f"   Best similarity: {score:.2%}, below: {args.thresh})")
            print(f"   See 'server_last_debug.png'")
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        print("🔌 Error: Failed to reach server.")
        sys.exit(2)

if __name__ == "__main__":
    main()