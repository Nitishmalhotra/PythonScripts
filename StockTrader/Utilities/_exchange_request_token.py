from pathlib import Path
from kiteconnect import KiteConnect

API_KEY = "u664cda77q2cf7ft"
API_SECRET = "krq6lxfi3mekiyyic0k5ubj60l9qui75"

def update_env_var(path, key, value):
    lines = []
    found = False
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                found = True
                break
    if not found:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    request_token = input("Enter request_token from login redirect: ").strip()
    if not request_token:
        print("ERROR: request_token is required")
        return

    kite = KiteConnect(api_key=API_KEY)
    data = kite.generate_session(request_token, api_secret=API_SECRET)

    access_token = data.get("access_token")
    user_id = data.get("user_id")

    project_root = Path(__file__).resolve().parents[1]
    cred_file = project_root / "kite_credentials.txt"
    env_file = project_root / ".env"

    cred_file.write_text(
        f"API_KEY={API_KEY}\n"
        f"ACCESS_TOKEN={access_token}\n"
        f"USER_ID={user_id}\n"
        f"GeneratedBy=script\n",
        encoding="utf-8"
    )

    update_env_var(env_file, "KITE_API_KEY", API_KEY)
    update_env_var(env_file, "KITE_ACCESS_TOKEN", access_token)

    print("SUCCESS: access token saved")

if __name__ == "__main__":
    main()
