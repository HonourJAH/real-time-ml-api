"""
Streams synthetic transaction events to the Real-Time ML API's WebSocket
endpoint, printing each scoring result as it arrives — the WebSocket
equivalent of the curl commands used to demo the REST-based projects in
this portfolio, since a persistent connection can't be demoed with a
single one-off curl call.

Usage:
    python3 scripts/simulate_stream.py
    python3 scripts/simulate_stream.py --anomaly-rate 0.15 --delay 0.5
    python3 scripts/simulate_stream.py --count 50 --host localhost --port 8000

Requires the API to already be running with a trained model
(POST /train first, or this will get a clear error back per event).
"""

import argparse
import asyncio
import json
import sys

import websockets

sys.path.insert(0, ".")
from app.services.event_generator import generate_event


async def stream(
    host: str, port: int, anomaly_rate: float, delay: float, count: int | None
):
    uri = f"ws://{host}:{port}/ws/stream"
    print(f"Connecting to {uri} ...")

    total = 0
    correct = 0
    latencies = []

    try:
        async with websockets.connect(uri) as ws:
            print("Connected. Streaming events (Ctrl+C to stop)\n")

            while count is None or total < count:
                event, is_actually_anomalous = generate_event(anomaly_rate)

                await ws.send(json.dumps(event))
                raw_response = await ws.recv()
                response = json.loads(raw_response)

                if "error" in response:
                    print(f"  ERROR: {response['error']}")
                    await asyncio.sleep(delay)
                    continue

                total += 1
                predicted_anomaly = response["is_anomaly"]
                match = predicted_anomaly == is_actually_anomalous
                correct += match

                latencies.append(response["latency_ms"])

                marker = "🚨 ANOMALY" if predicted_anomaly else "   normal "
                check = "✓" if match else "✗"
                print(
                    f"{marker}  score={response['anomaly_score']:+.4f}  "
                    f"latency={response['latency_ms']:6.2f}ms  "
                    f"(actual: {'anomaly' if is_actually_anomalous else 'normal':7s})  {check}  "
                    f"amount=${event['amount']:.2f}"
                )

                await asyncio.sleep(delay)

    except (ConnectionRefusedError, OSError):
        print(f"\nCould not connect to {uri}. Is the API running?")
        return
    except KeyboardInterrupt:
        pass
    finally:
        if total > 0:
            accuracy = correct / total * 100
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            print("\n--- Summary ---")
            print(f"Events streamed: {total}")
            print(f"Accuracy vs ground truth: {correct}/{total} ({accuracy:.1f}%)")
            print(f"Average latency: {avg_latency:.2f}ms")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        help="Probability each event is anomalous",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3, help="Seconds between events"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of events to send (default: run until Ctrl+C)",
    )
    args = parser.parse_args()

    asyncio.run(stream(args.host, args.port, args.anomaly_rate, args.delay, args.count))


if __name__ == "__main__":
    main()
