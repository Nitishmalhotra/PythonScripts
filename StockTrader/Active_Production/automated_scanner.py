"""
Automated Stock Scanner with Telegram Notifications
Runs scanner and sends alerts automatically
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path
from advanced_scanner import AdvancedStockScanner, get_strategy_recommendation
from telegram_notifier import TelegramNotifier
from enhanced_html_generator import generate_enhanced_html, calculate_market_sentiment
from dotenv import load_dotenv

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "Results"

# Ensure Results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'scanner_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_env_file(file_path):
    """Load environment variables from a .env-style file"""
    file_path = Path(file_path)
    if not file_path.exists():
        return
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_kite_credentials(project_root):
    """Load Kite credentials from kite_credentials.txt if available."""
    credentials_path = Path(project_root) / 'kite_credentials.txt'
    credentials = {}

    if not credentials_path.exists():
        return credentials

    with open(credentials_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                credentials[key] = value

    return credentials


class AutomatedScanner:
    """
    Automated scanner with Telegram notifications
    """
    
    def __init__(self, api_key, access_token, telegram_token, telegram_chat_id):
        """
        Initialize automated scanner
        
        Args:
            api_key (str): Kite API key
            access_token (str): Kite access token
            telegram_token (str): Telegram bot token
            telegram_chat_id (str): Telegram chat ID
        """
        self.scanner = AdvancedStockScanner(api_key, access_token)
        self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
        self.results = None  # Flat DataFrame for Telegram
        self.strategy_results = None  # Dict format for HTML
        
    def run_scan(self, stock_symbols=None):
        """
        Run the stock scanner
        
        Args:
            stock_symbols (list): List of stock symbols to scan
            
        Returns:
            DataFrame: Scan results
        """
        try:
            logger.info("🚀 Starting automated scan...")
            self.telegram.send_message("🚀 <b>Scanner Started</b>\n\nScanning Nifty 50 stocks...")
            
            # Default to Nifty 50 if no symbols provided
            if stock_symbols is None:
                stock_symbols = [
                    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
                    'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BHARTIARTL', 'BPCL',
                    'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
                    'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HEROMOTOCO',
                    'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'ITC',
                    'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI',
                    'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE',
                    'SBILIFE', 'SBIN', 'SHREECEM', 'SUNPHARMA', 'TATASTEEL',
                    'TATACONSUM', 'TATAMOTORS', 'TCS', 'TECHM', 'TITAN',
                    'ULTRACEMCO', 'UPL', 'WIPRO', 'INFY','BEL'
                ]
            
            # Load full instrument list and filter by symbols
            all_stocks = self.scanner.get_nse_stocks('NSE')
            
            def _normalize(sym):
                return ''.join(ch for ch in str(sym).upper() if ch.isalnum())
            
            symbol_set = set(_normalize(s) for s in stock_symbols)
            stocks_to_scan = [s for s in all_stocks if _normalize(s.get('tradingsymbol', '')) in symbol_set]
            
            if not stocks_to_scan:
                logger.warning("No symbols matched instrument list; falling back to first 50 instruments")
                stocks_to_scan = all_stocks[:50]
            
            logger.info(f"Scanning {len(stocks_to_scan)} instruments...")
            stock_data = self.scanner.scan_stocks_for_strategies(stocks_to_scan, lookback_days=300)
            
            if stock_data.empty:
                logger.warning("⚠️ No stock data retrieved")
                return pd.DataFrame()
            
            strategy_results = self.scanner.scan_with_strategies(stock_data)
            all_results = [df for df in strategy_results.values() if not df.empty]
            
            if not all_results:
                logger.warning("⚠️ No signals found")
                return pd.DataFrame()
            
            # Concatenate all results first
            self.results = pd.concat(all_results, ignore_index=True)  # Flat for processing
            
            # Filter to only recent signals (last 7 days) - keeps data relevant for today
            from datetime import timedelta
            self.results['date'] = pd.to_datetime(self.results['date'])
            # Make cutoff_date timezone-aware to match DataFrame dates
            cutoff_date = datetime.now() - timedelta(days=7)
            if self.results['date'].dt.tz is not None:
                # If dates are timezone-aware, localize cutoff_date to match
                cutoff_date = pd.Timestamp(cutoff_date).tz_localize(self.results['date'].dt.tz)
            initial_count = len(self.results)
            self.results = self.results[self.results['date'] >= cutoff_date].copy()
            filtered_count = initial_count - len(self.results)
            
            if filtered_count > 0:
                logger.info(f"📅 Filtered out {filtered_count} old signals (keeping last 7 days only)")
            
            # Create filtered strategy_results dict (same 7-day filter) for HTML
            self.strategy_results = {}
            for strategy_name, df in strategy_results.items():
                if not df.empty:
                    df_copy = df.copy()
                    df_copy['date'] = pd.to_datetime(df_copy['date'])
                    filtered_df = df_copy[df_copy['date'] >= cutoff_date]
                    if not filtered_df.empty:
                        self.strategy_results[strategy_name] = filtered_df
            
            if self.results.empty:
                logger.warning("⚠️ No recent signals found (all signals are older than 7 days)")
                return pd.DataFrame()
            
            # Ensure columns used by Telegram notifications exist
            # 1. risk_reward from rr_ratio_1
            if 'risk_reward' not in self.results.columns and 'rr_ratio_1' in self.results.columns:
                self.results['risk_reward'] = self.results['rr_ratio_1']
            
            # 2. action from strategy name
            if 'action' not in self.results.columns and 'strategy' in self.results.columns:
                self.results['action'] = self.results['strategy'].apply(get_strategy_recommendation)
            
            # 3. risk_level from risk_pct
            if 'risk_level' not in self.results.columns and 'risk_pct' in self.results.columns:
                def _risk_level(val):
                    try:
                        if pd.isna(val):
                            return 'MEDIUM'
                        if val <= 2:
                            return 'LOW'
                        if val <= 4:
                            return 'MEDIUM'
                        return 'HIGH'
                    except Exception:
                        return 'MEDIUM'
                self.results['risk_level'] = self.results['risk_pct'].apply(_risk_level)
            
            # 4. target from target_1
            if 'target' not in self.results.columns and 'target_1' in self.results.columns:
                self.results['target'] = self.results['target_1']
            
            # Log available columns for debugging
            logger.info(f"✅ Scan completed: {self.results['symbol'].nunique()} stocks, {len(self.results)} recent signals")
            logger.info(f"📅 Date range: {self.results['date'].min()} to {self.results['date'].max()}")
            logger.info(f"📋 Available columns: {self.results.columns.tolist()}")
            logger.info(f"🔍 Sample data:\n{self.results.head(2)}")
            return self.results
                
        except Exception as e:
            logger.error(f"❌ Scan failed: {e}")
            self.telegram.send_error_alert(f"Scanner failed: {str(e)}")
            raise
    
    def send_notifications(self, send_csv=False, send_high_priority=True):
        """
        Send Telegram notifications
        
        Args:
            send_csv (bool): Send CSV file
            send_high_priority (bool): Send high-priority signals
        """
        if self.results is None or self.results.empty:
            self.telegram.send_message("ℹ️ No trading signals found in this scan")
            return
        
        try:
            # IMPORTANT: Replicate HTML deduplication logic exactly
            # Build list of all signals
            all_signals = []
            for _, row in self.results.iterrows():
                signal_date = row.get('date')
                if signal_date is None or (isinstance(signal_date, float) and pd.isna(signal_date)):
                    continue
                normalized_date = pd.to_datetime(signal_date).normalize()
                all_signals.append({
                    'symbol': row['symbol'],
                    'strategy': row['strategy'],
                    'date': signal_date,
                    'normalized_date': normalized_date,
                    'action': row.get('action', 'HOLD'),
                    'row_data': row
                })
            
            # Deduplicate: keep LATEST signal per stock (same as HTML)
            stock_latest = {}
            for signal in all_signals:
                symbol = signal['symbol']
                if symbol not in stock_latest:
                    stock_latest[symbol] = signal
                else:
                    if signal['normalized_date'] > stock_latest[symbol]['normalized_date']:
                        stock_latest[symbol] = signal
            
            # Rebuild DataFrame from deduplicated signals
            latest_per_stock = pd.DataFrame([sig['row_data'] for sig in stock_latest.values()])
            
            # Calculate summary stats from deduplicated data
            unique_stocks = len(stock_latest)
            strategies = latest_per_stock['strategy'].unique()
            active_strategies = len(strategies)
            
            # Get sentiment from deduplicated data
            latest_dict = {
                row['symbol']: {'recommendation': row.get('action', 'HOLD')}
                for _, row in latest_per_stock.iterrows()
            }
            sentiment_label, sentiment_pct = calculate_market_sentiment(latest_dict)
            sentiment = f"{sentiment_label} {sentiment_pct:.0f}%"
            
            # Strategy counts from deduplicated data (matches HTML)
            strategy_counts = latest_per_stock.groupby('strategy').size()
            top_strategy = strategy_counts.idxmax()
            top_strategy_count = strategy_counts.max()
            
            # 1. Send summary
            logger.info("📤 Sending summary notification...")
            self.telegram.send_scanner_summary(
                total_stocks=unique_stocks,
                active_strategies=active_strategies,
                sentiment=sentiment,
                top_strategy=(top_strategy, top_strategy_count)
            )
            
            # 2. Send high-priority alerts (from deduplicated data)
            if send_high_priority:
                logger.info("📤 Sending high-priority alerts...")
                self.telegram.send_high_priority_alerts(latest_per_stock)
            
            # 3. Send strategy breakdown (from deduplicated counts)
            logger.info("📤 Sending strategy breakdown...")
            self.telegram.send_strategy_breakdown(dict(strategy_counts))
            
            # 4. Send top signals for top 3 strategies (from deduplicated data)
            logger.info("📤 Sending top strategy signals...")
            top_3_strategies = strategy_counts.nlargest(3)
            
            for strategy in top_3_strategies.index:
                strategy_signals = latest_per_stock[latest_per_stock['strategy'] == strategy]
                if not strategy_signals.empty:
                    self.telegram.send_top_signals(strategy_signals, strategy, max_signals=5)
            
            # 5. Send CSV file if requested
            if send_csv:
                csv_filename = RESULTS_DIR / f"strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.results.to_csv(csv_filename, index=False)
                logger.info(f"📤 Sending CSV file: {csv_filename}")
                self.telegram.send_file(
                    csv_filename,
                    caption=f"📊 Scanner Results - {unique_stocks} stocks, {len(latest_per_stock)} latest signals"
                )
            
            logger.info("✅ All notifications sent successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to send notifications: {e}")
            self.telegram.send_error_alert(f"Notification failed: {str(e)}")
    
    def generate_html_report(self):
        """
        Generate HTML dashboard
        
        Returns:
            str: Path to generated HTML file
        """
        if self.results is None or self.results.empty:
            logger.warning("⚠️ No results to generate HTML")
            return None
        
        try:
            logger.info("📄 Generating HTML dashboard...")
            
            csv_filename = RESULTS_DIR / f"strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.results.to_csv(csv_filename, index=False)
            html_output_file = RESULTS_DIR / "scanner_results.html"
            
            # Use strategy_results dict for HTML (shows latest signal per stock)
            # Suppress all terminal output during HTML generation
            import warnings
            import io
            import contextlib
            
            with warnings.catch_warnings(), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                html_content = generate_enhanced_html(
                    self.strategy_results,
                    datetime.now().strftime('%Y%m%d_%H%M%S')
                )

            with open(html_output_file, 'w', encoding='utf-8') as file:
                file.write(html_content)
            
            if html_output_file.exists():
                logger.info(f"✅ HTML dashboard generated: {html_output_file}")
            else:
                logger.warning("⚠️ HTML generation returned None")
            
            return str(html_output_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate HTML: {e}")
            return None
    

    
    def run_full_automation(self, send_csv=False, generate_html=True):
        """
        Run complete automation: scan + notify + report
        
        Args:
            send_csv (bool): Send CSV via Telegram
            generate_html (bool): Generate HTML dashboard
        """
        try:
            start_time = datetime.now()
            logger.info(f"{'='*60}")
            logger.info(f"🤖 AUTOMATED SCANNER STARTED - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            # 1. Run scan
            results = self.run_scan()
            
            # 2. Generate HTML report first (independent of Telegram state)
            if generate_html and not results.empty:
                html_file = self.generate_html_report()
                if html_file:
                    logger.info(f"📊 Dashboard available at: {html_file}")

            # 3. Send notifications
            if not results.empty:
                self.send_notifications(send_csv=send_csv, send_high_priority=True)
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"{'='*60}")
            logger.info(f"✅ AUTOMATION COMPLETED in {duration:.2f} seconds")
            logger.info(f"{'='*60}")
            
            # Final summary to Telegram
            if not results.empty:
                self.telegram.send_message(
                    f"✅ <b>Scan Complete</b>\n\n"
                    f"Duration: {duration:.1f}s\n"
                    f"Signals: {len(results)}\n"
                    f"Unique Stocks: {results['symbol'].nunique()}"
                )
            
        except Exception as e:
            logger.error(f"❌ Automation failed: {e}")
            self.telegram.send_error_alert(f"Automation failed: {str(e)}")
            raise


def main():
    """
    Main entry point for automated scanner
    """
    # Load .env files from project root (override to avoid stale terminal env values)
    load_dotenv(PROJECT_ROOT / '.env', override=True)
    load_dotenv(PROJECT_ROOT / '.env.telegram', override=True)

    # Load Kite credentials with precedence: kite_credentials.txt > environment
    kite_credentials = load_kite_credentials(PROJECT_ROOT)
    API_KEY = kite_credentials.get('API_KEY') or os.getenv('KITE_API_KEY')
    ACCESS_TOKEN = kite_credentials.get('ACCESS_TOKEN') or os.getenv('KITE_ACCESS_TOKEN')
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Validate credentials
    if not all([API_KEY, ACCESS_TOKEN, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID]):
        logger.error("❌ Missing credentials! Set environment variables:")
        logger.error("   - KITE_API_KEY")
        logger.error("   - KITE_ACCESS_TOKEN")
        logger.error("   - TELEGRAM_BOT_TOKEN")
        logger.error("   - TELEGRAM_CHAT_ID")
        sys.exit(1)
    
    try:
        # Initialize and run
        scanner = AutomatedScanner(
            api_key=API_KEY,
            access_token=ACCESS_TOKEN,
            telegram_token=TELEGRAM_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID
        )
        
        # Run full automation
        scanner.run_full_automation(
            send_csv=True,      # Send CSV file to Telegram
            generate_html=True  # Generate HTML dashboard
        )
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
