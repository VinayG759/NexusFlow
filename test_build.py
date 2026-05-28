
import httpx
import asyncio

BASE_URL = "http://localhost:8000"

async def main():
    """POST /build, then poll /build/status/{job_id} until done."""
    problem_statement = (
        "Create a simple web page with a title that says 'Hello World' "
        "and a button that shows an alert when clicked."
    )

    print("Sending build request...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{BASE_URL}/build",
                json={"problem_statement": problem_statement},
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            print(f"Job started — id={job_id}")

        # Poll until terminal state
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                resp = await client.get(f"{BASE_URL}/build/status/{job_id}")
                resp.raise_for_status()
                data = resp.json()
                status = data["status"]
                print(f"  [{status}] {data.get('progress', '')}")

                if status == "complete":
                    r = data["result"]
                    print(f"\nBuild complete!")
                    print(f"  Project : {r['project_name']}")
                    print(f"  Files   : {r['total_files']}")
                    print(f"  Fixes   : {r['debug_fixes_applied']}")
                    print(f"  Backend : {'OK' if r['backend_verified'] else 'FAIL'}")
                    print(f"  Frontend: {'OK' if r['frontend_verified'] else 'FAIL'}")
                    if r.get("debug_remaining_errors"):
                        print(f"  Remaining errors: {r['debug_remaining_errors']}")
                    break
                elif status == "failed":
                    print(f"\nBuild failed: {data.get('error', 'unknown error')}")
                    break

                await asyncio.sleep(10)

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code} — {e.response.text}")
    except httpx.RequestError as e:
        print(f"Request error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
