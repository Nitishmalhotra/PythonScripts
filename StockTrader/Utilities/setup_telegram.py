"""
Quick Setup Script for Telegram Bot
Helps you configure the bot and test it
"""

import os
import sys
import requests


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def get_bot_token():
    """Get bot token from user"""
    print_header("Step 1: Get Your Bot Token")
    print("📱 Instructions:")
    print("1. Open Telegram and search for @BotFather")
    print("2. Send /newbot command")
    print("3. Follow the prompts to create your bot")
    print("4. Copy the bot token\n")
    
    token = input("Paste your bot token here: ").strip()
    
    if not token or ':' not in token:
        print("❌ Invalid token format. Should look like: 1234567890:ABCdef...")
        sys.exit(1)
    
    return token


def get_chat_id(bot_token):
    """Get chat ID using bot token"""
    print_header("Step 2: Get Your Chat ID")
    print("📱 Instructions:")
    print("1. Search for your bot in Telegram")
    print("2. Start a chat and send any message (e.g., 'hello')")
    input("\nPress Enter after you've sent a message to your bot...")
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['result']:
                chat_ids = []
                for update in data['result']:
                    if 'message' in update:
                        chat = update['message']['chat']
                        chat_ids.append({
                            'id': chat['id'],
                            'type': chat['type'],
                            'name': chat.get('first_name', chat.get('username', 'Unknown'))
                        })
                
                if chat_ids:
                    print("\n✅ Found chat(s):")
                    for i, chat in enumerate(chat_ids, 1):
                        print(f"{i}. Chat ID: {chat['id']} (Type: {chat['type']}, Name: {chat['name']})")
                    
                    if len(chat_ids) == 1:
                        return str(chat_ids[0]['id'])
                    else:
                        choice = input(f"\nSelect chat (1-{len(chat_ids)}): ").strip()
                        return str(chat_ids[int(choice)-1]['id'])
                else:
                    print("❌ No chats found. Make sure you sent a message to your bot!")
                    sys.exit(1)
            else:
                print("❌ No messages found. Please send a message to your bot first!")
                sys.exit(1)
        else:
            print(f"❌ Error: {response.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def test_bot(bot_token, chat_id):
    """Send test message"""
    print_header("Step 3: Testing Bot")
    print("Sending test message...")
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': '✅ <b>Success!</b>\n\nYour Telegram bot is working!\n\n🤖 Stock Scanner Bot is ready to send alerts.',
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("✅ Test message sent! Check your Telegram.")
            return True
        else:
            print(f"❌ Failed to send message: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def create_env_file(bot_token, chat_id):
    """Create .env file with credentials"""
    print_header("Step 4: Creating Configuration File")
    
    # Check for existing Kite credentials
    kite_api_key = os.getenv('KITE_API_KEY', '')
    kite_access_token = os.getenv('KITE_ACCESS_TOKEN', '')
    
    if not kite_api_key:
        print("Enter your Kite Connect credentials:")
        kite_api_key = input("Kite API Key: ").strip()
    
    if not kite_access_token:
        kite_access_token = input("Kite Access Token: ").strip()
    
    env_content = f"""# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN={bot_token}
TELEGRAM_CHAT_ID={chat_id}

# Kite Connect Credentials
KITE_API_KEY={kite_api_key}
KITE_ACCESS_TOKEN={kite_access_token}
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Configuration saved to .env file")


def create_load_env_script():
    """Create script to load environment variables"""
    content = """# Load Environment Variables from .env file
# Source this in PowerShell: . .\\load_env.ps1

Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
        Write-Host "Set $name"
    }
}

Write-Host "`n✅ Environment variables loaded!"
"""
    
    with open('load_env.ps1', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Created load_env.ps1 script")


def main():
    """Main setup flow"""
    print_header("🤖 Telegram Bot Setup Wizard")
    print("This wizard will help you set up your Telegram bot for stock alerts.\n")
    
    # Check if requests is installed
    try:
        import requests
    except ImportError:
        print("❌ 'requests' package not found. Installing...")
        os.system("pip install requests")
        import requests
    
    # Get credentials
    bot_token = get_bot_token()
    chat_id = get_chat_id(bot_token)
    
    # Test the bot
    if test_bot(bot_token, chat_id):
        # Create config files
        create_env_file(bot_token, chat_id)
        create_load_env_script()
        
        print_header("✅ Setup Complete!")
        print("Your Telegram bot is configured and ready!\n")
        print("📋 Next Steps:")
        print("1. Test the automated scanner:")
        print("   python automated_scanner.py\n")
        print("2. Schedule automated scans:")
        print("   See TELEGRAM_SETUP_GUIDE.md for instructions\n")
        print("3. To load environment variables in PowerShell:")
        print("   . .\\load_env.ps1\n")
        print("🎉 Happy Trading!")
    else:
        print("\n❌ Setup failed. Please check your credentials and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
