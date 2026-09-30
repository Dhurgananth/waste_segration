#!/usr/bin/env python3
import sys
import requests

API_URL = "http://localhost:8000/predict"

def test_image(image_path):
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        response = requests.post(API_URL, files=files)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        return
    
    data = response.json()
    
    print(f"\n{'='*50}")
    print(f"🎯 PREDICTION: {data['prediction'].upper()}")
    print(f"   Confidence: {data['confidence']}%")
    print(f"   Reliable: {'✅ Yes' if data['reliable'] else '⚠️ No - retake photo'}")
    print(f"\n🗑️  Bin: {data['disposal']['bin']}")
    print(f"💡 Tip: {data['disposal']['tip']}")
    print(f"\nTop 3:")
    for item in data["top_3"]:
        bar = "█" * int(item["confidence"] / 4)
        print(f"   {item['class']:12s} {item['confidence']:5.1f}% {bar}")
    print(f"{'='*50}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py path/to/photo.jpg")
        sys.exit(1)
    test_image(sys.argv[1])