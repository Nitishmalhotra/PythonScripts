"""
Kite Connect - Access Token Generator
This script helps you generate an access token for the Kite Connect API
"""

from pathlib import Path
from kiteconnect import KiteConnect


def upsert_env_value(env_path, key, value):
    """Insert or update KEY=VALUE in .env style file."""
    env_file = Path(env_path)
    lines = []

    if env_file.exists():
        lines = env_file.read_text(encoding='utf-8').splitlines()

    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        if new_lines and new_lines[-1].strip() != "":
            new_lines.append("")
        new_lines.append(f"{key}={value}")

    env_file.write_text("\n".join(new_lines) + "\n", encoding='utf-8')

def generate_access_token():
    """
    Interactive script to generate Kite Connect access token
    """
    print("="*70)
    print("KITE CONNECT - ACCESS TOKEN GENERATOR")
    print("="*70)
    print()
    
    # API credentials
    api_key = "u664cda77q2cf7ft"
    api_secret = "krq6lxfi3mekiyyic0k5ubj60l9qui75"
    
    print(f"Using API Key: {api_key}")
    print(f"Using API Secret: {api_secret[:10]}...")  # Show only first 10 chars for security
    
    if not api_key or not api_secret:
        print("Error: API Key and Secret are required!")
        return
    
    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    
    # Generate login URL
    login_url = kite.login_url()
    
    print("\n" + "="*70)
    print("STEP 1: LOGIN TO KITE")
    print("="*70)
    print("\n1. Open the following URL in your browser:")
    print(f"\n{login_url}\n")
    print("2. Login with your Zerodha credentials")
    print("3. After successful login, you'll be redirected to a URL")
    print("4. Copy the 'request_token' from the redirected URL")
    print("\n   Example: https://your-redirect-url?request_token=ABC123XYZ")
    print("   Copy: ABC123XYZ")
    print()
    
    # Get request token from user
    request_token = input("Enter the request token: ").strip()
    
    if not request_token:
        print("Error: Request token is required!")
        return
    
    try:
        # Generate session
        print("\nGenerating access token...")
        data = kite.generate_session(request_token, api_secret=api_secret)
        
        print("\n" + "="*70)
        print("SUCCESS! ACCESS TOKEN GENERATED")
        print("="*70)
        print(f"\nAccess Token: {data['access_token']}")
        print(f"User ID: {data['user_id']}")
        print(f"User Name: {data['user_name']}")
        print(f"\nPublic Token: {data['public_token']}")
        print()
        print("="*70)
        print("IMPORTANT NOTES:")
        print("="*70)
        print("1. Save this access token - you'll need it to run the scanner")
        print("2. Access tokens expire at the end of each day (3:30 PM)")
        print("3. You'll need to generate a new token daily")
        print("4. Update your kite_stock_scanner.py with this token")
        print()
        print("Add this to your script:")
        print(f'ACCESS_TOKEN = "{data["access_token"]}"')
        print("="*70)
        
        # Save to file
        save_option = input("\nDo you want to save credentials to a file? (yes/no): ").strip().lower()
        if save_option == 'yes':
            with open('kite_credentials.txt', 'w') as f:
                f.write(f"API_KEY={api_key}\n")
                f.write(f"ACCESS_TOKEN={data['access_token']}\n")
                f.write(f"USER_ID={data['user_id']}\n")
                f.write(f"Generated on: {data.get('login_time', 'N/A')}\n")
            print("\nCredentials saved to 'kite_credentials.txt'")

            # Also sync .env used by automated_scanner.py
            project_env = Path(__file__).resolve().parents[1] / '.env'
            upsert_env_value(project_env, 'KITE_API_KEY', api_key)
            upsert_env_value(project_env, 'KITE_ACCESS_TOKEN', data['access_token'])
            print(f"Kite credentials synced to '{project_env}'")
            print("⚠️  Remember: Keep this file secure and don't share it!")
        
    except Exception as e:
        # Show detailed exception information to help debugging
        print(f"\nError generating access token: {repr(e)}")
        print("\nCommon issues:")
        print("1. Invalid request token - make sure you copied it exactly from the redirected URL (no extra chars)")
        print("2. Request token already used or expired - generate a new one immediately after login")
        print("3. API Key/Secret mismatch - ensure the API keay/secret belong to your Kite app")
        print("4. System clock skew or network issues can also cause failures")


if __name__ == "__main__":
    try:
        generate_access_token()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")