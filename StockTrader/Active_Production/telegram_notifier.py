"""
Telegram Notification Module
Sends trading alerts and scanner results to Telegram
"""

import requests
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Send notifications to Telegram bot
    """
    
    def __init__(self, bot_token, chat_id):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token (str): Your Telegram bot token from @BotFather
            chat_id (str): Your Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_message(self, message, parse_mode='HTML'):
        """
        Send text message to Telegram
        
        Args:
            message (str): Message text (supports HTML formatting)
            parse_mode (str): Message formatting (HTML or Markdown)
            
        Returns:
            bool: True if sent successfully
        """
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("✅ Telegram message sent successfully")
                return True
            else:
                logger.error(f"❌ Telegram error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram message: {e}")
            return False
    
    def send_scanner_summary(self, total_stocks, active_strategies, sentiment, top_strategy):
        """
        Send scanner summary notification
        
        Args:
            total_stocks (int): Total stocks scanned
            active_strategies (int): Number of active strategies
            sentiment (str): Market sentiment (BULLISH/BEARISH)
            top_strategy (tuple): (strategy_name, count)
        """
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        sentiment_emoji = "🟢" if "BULLISH" in sentiment else "🔴"
        
        message = f"""
📊 <b>Stock Scanner Alert</b>
⏰ {timestamp}

📈 <b>Scan Results:</b>
• Total Stocks: <b>{total_stocks}</b>
• Active Strategies: <b>{active_strategies}</b>
• Market Sentiment: {sentiment_emoji} <b>{sentiment}</b>
• Top Strategy: <b>{top_strategy[0]}</b> ({top_strategy[1]} stocks)

🔗 Check dashboard for details
        """
        
        return self.send_message(message.strip())
    
    def send_top_signals(self, signals_df, strategy_name, max_signals=5):
        """
        Send top trading signals for a strategy
        
        Args:
            signals_df (DataFrame): Signals dataframe
            strategy_name (str): Strategy name
            max_signals (int): Maximum number of signals to send
        """
        if signals_df.empty:
            return
                # Validate required columns exist
        required_cols = ['risk_reward', 'action', 'symbol', 'close', 'stop_loss', 'target', 'rsi_14', 'volume_ratio']
        missing_cols = [col for col in required_cols if col not in signals_df.columns]
        
        if missing_cols:
            logger.warning(f"⚠️ Top signals for {strategy_name} skipped - missing columns: {missing_cols}")
            return
                # Sort by risk-reward ratio and take top signals
        top_signals = signals_df.nlargest(max_signals, 'risk_reward')
        
        message = f"📌 <b>{strategy_name}</b>\n\n"
        
        for idx, row in top_signals.iterrows():
            action_emoji = "🟢" if row['action'] == 'BUY' else "🔴" if row['action'] == 'SELL' else "🟡"
            
            message += f"{action_emoji} <b>{row['symbol']}</b>\n"
            message += f"  Price: ₹{row['close']:.2f}\n"
            message += f"  Action: {row['action']}\n"
            message += f"  R:R: {row['risk_reward']:.2f}\n"
            message += f"  SL: ₹{row['stop_loss']:.2f} | Target: ₹{row['target']:.2f}\n"
            message += f"  RSI: {row['rsi_14']:.1f} | Vol: {row['volume_ratio']:.2f}x\n\n"
        
        return self.send_message(message.strip())
    
    def send_high_priority_alerts(self, signals_df):
        """
        Send high-priority signals (low risk, high R:R)
        
        Args:
            signals_df (DataFrame): All signals dataframe
        """
        # Validate required columns exist
        required_cols = ['risk_level', 'risk_reward', 'volume_ratio', 'action', 'symbol', 'strategy', 'close', 'stop_loss', 'target', 'rsi_14']
        missing_cols = [col for col in required_cols if col not in signals_df.columns]
        
        if missing_cols:
            logger.warning(f"⚠️ High-priority alerts skipped - missing columns: {missing_cols}")
            return
        
        # Filter high-quality setups
        priority_signals = signals_df[
            (signals_df['risk_level'] == 'LOW') & 
            (signals_df['risk_reward'] >= 1.67) &
            (signals_df['volume_ratio'] > 1.5)
        ].head(10)
        
        if priority_signals.empty:
            return
        
        message = "🔥 <b>HIGH PRIORITY SIGNALS</b> 🔥\n\n"
        
        for idx, row in priority_signals.iterrows():
            action_emoji = "🟢" if row['action'] == 'BUY' else "🔴"
            
            message += f"{action_emoji} <b>{row['symbol']}</b> - {row['strategy']}\n"
            message += f"  ₹{row['close']:.2f} | R:R {row['risk_reward']:.2f} | RSI {row['rsi_14']:.1f}\n"
            message += f"  SL: ₹{row['stop_loss']:.2f} → Target: ₹{row['target']:.2f}\n\n"
        
        return self.send_message(message.strip())
    
    def send_strategy_breakdown(self, strategy_counts):
        """
        Send breakdown of signals by strategy
        
        Args:
            strategy_counts (dict): Dictionary of strategy names and counts
        """
        message = "📊 <b>Strategy Breakdown</b>\n\n"
        
        # Sort by count descending
        sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
        
        for strategy, count in sorted_strategies:
            if count > 0:
                message += f"• <b>{strategy}</b>: {count} stocks\n"
        
        return self.send_message(message.strip())
    
    def send_file(self, file_path, caption=None):
        """
        Send file (CSV, HTML, etc.) to Telegram
        
        Args:
            file_path (str): Path to file
            caption (str): Optional caption
            
        Returns:
            bool: True if sent successfully
        """
        try:
            url = f"{self.base_url}/sendDocument"
            
            with open(file_path, 'rb') as file:
                files = {'document': file}
                data = {'chat_id': self.chat_id}
                
                if caption:
                    data['caption'] = caption
                
                response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                logger.info(f"✅ File sent successfully: {file_path}")
                return True
            else:
                logger.error(f"❌ Failed to send file: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to send file: {e}")
            return False
    
    def send_error_alert(self, error_message):
        """
        Send error notification
        
        Args:
            error_message (str): Error description
        """
        message = f"""
⚠️ <b>Scanner Error Alert</b>

❌ {error_message}

Time: {datetime.now().strftime("%I:%M %p")}
        """
        
        return self.send_message(message.strip())


def get_telegram_chat_id(bot_token):
    """
    Helper function to get your Telegram chat ID
    
    Steps:
    1. Start a chat with your bot
    2. Send any message
    3. Run this function to see your chat ID
    
    Args:
        bot_token (str): Your bot token
        
    Returns:
        str: Instructions and chat IDs
    """
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['result']:
                print("📱 Found the following chat IDs:")
                print("-" * 50)
                
                for update in data['result']:
                    if 'message' in update:
                        chat = update['message']['chat']
                        print(f"Chat ID: {chat['id']}")
                        print(f"Type: {chat['type']}")
                        if 'username' in chat:
                            print(f"Username: @{chat['username']}")
                        if 'first_name' in chat:
                            print(f"Name: {chat['first_name']}")
                        print("-" * 50)
                
                return "Use the Chat ID shown above in your config"
            else:
                return "No messages found. Please send a message to your bot first!"
        else:
            return f"Error: {response.text}"
            
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    """
    Test the Telegram notifier
    """
    import os
    
    # Load from environment or config
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
    CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
    
    if BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print("❌ Please set TELEGRAM_BOT_TOKEN environment variable")
        print("\nTo get your chat ID, run:")
        print("  python telegram_notifier.py --get-chat-id YOUR_BOT_TOKEN")
    else:
        notifier = TelegramNotifier(BOT_TOKEN, CHAT_ID)
        
        # Test message
        success = notifier.send_message("✅ Telegram notifier is working!")
        
        if success:
            print("✅ Test message sent successfully!")
        else:
            print("❌ Failed to send test message")
