"""
Enhanced HTML Generator for Advanced Stock Scanner
Includes: Dark Mode, Dashboard Summary, Export Functions, Risk Badges, Enhanced Data Display
"""

import pandas as pd
from datetime import datetime

def calculate_market_sentiment(stock_strategies):
    """Calculate overall market sentiment from stock signals"""
    if not stock_strategies:
        return "NEUTRAL", 50
    
    buy_count = sum(1 for data in stock_strategies.values() if data['recommendation'] == 'BUY')
    sell_count = sum(1 for data in stock_strategies.values() if data['recommendation'] == 'SELL')
    total = len(stock_strategies)
    
    buy_pct = (buy_count / total * 100) if total > 0 else 50
    
    if buy_pct >= 70:
        return "BULLISH", buy_pct
    elif buy_pct >= 55:
        return "MODERATELY BULLISH", buy_pct
    elif buy_pct >= 45:
        return "NEUTRAL", buy_pct
    elif buy_pct >= 30:
        return "MODERATELY BEARISH", buy_pct
    else:
        return "BEARISH", buy_pct

def calculate_risk_level(rr_ratio, volume_ratio, rsi):
    """Calculate risk level based on R:R ratio, volume, and RSI"""
    risk_score = 0
    
    # R:R ratio contribution (0-40 points)
    if rr_ratio >= 3:
        risk_score += 40
    elif rr_ratio >= 2:
        risk_score += 30
    elif rr_ratio >= 1.5:
        risk_score += 20
    else:
        risk_score += 10
    
    # Volume contribution (0-30 points)
    if volume_ratio >= 1.5:
        risk_score += 30
    elif volume_ratio >= 1.2:
        risk_score += 20
    else:
        risk_score += 10
    
    # RSI contribution (0-30 points)
    if 40 <= rsi <= 60:
        risk_score += 30
    elif 30 <= rsi <= 70:
        risk_score += 20
    else:
        risk_score += 10
    
    # Determine risk level
    if risk_score >= 75:
        return "LOW", "#28a745"  # Green
    elif risk_score >= 50:
        return "MEDIUM", "#ffc107"  # Yellow
    else:
        return "HIGH", "#dc3545"  # Red

def generate_enhanced_html(strategy_results, timestamp, stock_data_dict=None):
    """
    Generate professional HTML with all enhancement features
    stock_data_dict: Optional dict with additional data like {symbol: {'52w_high': val, '52w_low': val, 'atr': val}}
    """
    # Prepare data (same as original)
    all_stock_signals = []
    for strategy_name, df in strategy_results.items():
        if not df.empty:
            for _, row in df.iterrows():
                signal_date = row.get('date')
                if signal_date is None or (isinstance(signal_date, float) and pd.isna(signal_date)):
                    continue
                normalized_date = pd.to_datetime(signal_date).normalize()
                
                # Get additional data if available
                symbol = row['symbol']
                extra_data = stock_data_dict.get(symbol, {}) if stock_data_dict else {}
                
                all_stock_signals.append({
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'date': signal_date,
                    'normalized_date': normalized_date,
                    'close': row.get('close', 0),
                    'rsi_14': row.get('rsi_14', 0),
                    'volume_ratio': row.get('volume_ratio', 0),
                    'stop_loss': row.get('stop_loss', 0),
                    'target_1': row.get('target_1', 0),
                    'rr_ratio_1': row.get('rr_ratio_1', 0),
                    'atr': extra_data.get('atr', row.get('atr', 0)),
                    '52w_high': extra_data.get('52w_high', 0),
                    '52w_low': extra_data.get('52w_low', 0),
                    'action': get_strategy_recommendation(strategy_name)
                })
    
    # Allocate each stock to LATEST strategy
    stock_strategies = {}
    for signal in all_stock_signals:
        symbol = signal['symbol']
        if symbol not in stock_strategies:
            stock_strategies[symbol] = signal
        else:
            if signal['normalized_date'] > stock_strategies[symbol]['normalized_date']:
                stock_strategies[symbol] = signal
    
    # Update latest_date and calculate additional metrics
    for symbol, data in stock_strategies.items():
        data['latest_date'] = data['normalized_date'].strftime('%Y-%m-%d')
        data['strategies'] = [data['strategy']]
        data['recommendation'] = data['action']
        data['conflict'] = False
        
        # Calculate risk level
        risk_level, risk_color = calculate_risk_level(
            data['rr_ratio_1'],
            data['volume_ratio'],
            data['rsi_14']
        )
        data['risk_level'] = risk_level
        data['risk_color'] = risk_color
        
        # Calculate distance from 52-week high
        if data['52w_high'] > 0:
            data['dist_from_52w_high'] = ((data['close'] - data['52w_high']) / data['52w_high']) * 100
        else:
            data['dist_from_52w_high'] = 0
    
    # Strategy summary
    strategy_stock_counts = {}
    for symbol, data in stock_strategies.items():
        strategy = data['strategy']
        if strategy not in strategy_stock_counts:
            strategy_stock_counts[strategy] = 0
        strategy_stock_counts[strategy] += 1
    
    strategy_summary = sorted(
        [{'name': name, 'count': count} for name, count in strategy_stock_counts.items()],
        key=lambda x: x['count'],
        reverse=True
    )
    
    # Calculate market sentiment
    sentiment, sentiment_pct = calculate_market_sentiment(stock_strategies)
    
    # Top 3 strategies
    top_3_strategies = strategy_summary[:3]
    
    html_content =f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Stock Scanner - Professional Dashboard</title>
    <style>
        :root {{
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-gradient: linear-gradient(135deg, #2a5298 0%, #20c997 100%);
            --text-primary: #333;
            --text-secondary: #6c757d;
            --border-color: #e0e0e0;
            --card-bg: #ffffff;
            --header-bg: linear-gradient(135deg, #28a745 0%, #333 100%);
            --table-header-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        [data-theme="dark"] {{
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-gradient: linear-gradient(135deg, #1a2940 0%, #0f5132 100%);
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --border-color: #444;
            --card-bg: #2d2d2d;
            --header-bg: linear-gradient(135deg, #1e6b3a 0%, #222 100%);
            --table-header-bg: linear-gradient(135deg, #4a5ba0 0%, #5a3d70 100%);
            --shadow: 0 2px 5px rgba(0,0,0,0.5);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
            transition: all 0.3s ease;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: var(--bg-primary);
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: var(--header-bg);
            color: white;
            padding: 30px;
            text-align: center;
            border-bottom: 5px solid #1e3c72;
            position: relative;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .theme-toggle {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255,255,255,0.2);
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.2em;
            transition: all 0.3s;
        }}
        
        .theme-toggle:hover {{
            background: rgba(255,255,255,0.3);
            transform: scale(1.05);
        }}
        
        .dashboard-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            padding: 30px;
            background: var(--bg-secondary);
        }}
        
        .summary-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: var(--shadow);
            border-left: 5px solid #667eea;
            transition: transform 0.2s;
        }}
        
        .summary-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }}
        
        .summary-label {{
            font-size: 0.9em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .summary-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: var(--text-primary);
        }}
        
        .summary-subtext {{
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-top: 5px;
        }}
        
        .sentiment-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .sentiment-bullish {{ background: #28a745; color: white; }}
        .sentiment-bearish {{ background: #dc3545; color: white; }}
        .sentiment-neutral {{ background: #ffc107; color: #333; }}
        
        .export-toolbar {{
            padding: 20px 30px;
            background: var(--bg-secondary);
            border-bottom: 2px solid var(--border-color);
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .btn {{
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            font-size: 0.9em;
        }}
        
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .search-box {{
            padding: 10px 20px;
            border: 2px solid var(--border-color);
            border-radius: 25px;
            font-size: 0.9em;
            background: var(--card-bg);
            color: var(--text-primary);
            flex-grow: 1;
            max-width: 400px;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: var(--text-primary);
        }}
        
        .collapse-btn {{
            background: none;
            border: none;
            font-size: 1.5em;
            cursor: pointer;
            color: var(--text-secondary);
            transition: transform 0.3s;
        }}
        
        .collapse-btn.collapsed {{
            transform: rotate(-90deg);
        }}
        
        .collapsible-content {{
            max-height: 2000px;
            overflow: hidden;
            transition: max-height 0.5s ease;
        }}
        
        .collapsible-content.collapsed {{
            max-height: 0;
        }}
        
        .tabs {{
            display: flex;
            border-bottom: 2px solid var(--border-color);
            margin-bottom: 20px;
            overflow-x: auto;
        }}
        
        .tab {{
            padding: 15px 25px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1em;
            font-weight: 600;
            color: var(--text-secondary);
            transition: all 0.3s;
            white-space: nowrap;
            border-bottom: 3px solid transparent;
        }}
        
        .tab:hover {{
            color: #667eea;
        }}
        
        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }}
        
        th {{
            background: var(--table-header-bg);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}
        
        tr:hover {{
            background: var(--bg-secondary);
        }}
        
        .risk-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.8em;
            text-align: center;
            display: inline-block;
        }}
        
        .price-change-positive {{
            color: #28a745;
            font-weight: 600;
        }}
        
        .price-change-negative {{
            color: #dc3545;
            font-weight: 600;
        }}
        
        .volume-highlight {{
            background: #fff3cd;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: 600;
        }}
        
        .connection-status {{
            position: absolute;
            top: 20px;
            left: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.2);
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .status-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }}
        
        .status-live {{
            background: #28a745;
        }}
        
        .status-cached {{
            background: #dc3545;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        
        .refresh-btn {{
            background: linear-gradient(135deg, #20c997 0%, #28a745 100%);
        }}
        
        .refresh-btn:hover {{
            background: linear-gradient(135deg, #1ba87e 0%, #218838 100%);
        }}
        
        .action-buy {{
            background: #28a745;
            color: white;
            padding: 5px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.85em;
        }}
        
        .action-sell {{
            background: #dc3545;
            color: white;
            padding: 5px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.85em;
        }}
        
        .action-hold {{
            background: #ffc107;
            color: #333;
            padding: 5px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.85em;
        }}
        
        .sparkline {{
            width: 100px;
            height: 30px;
        }}
        
        @media print {{
            .export-toolbar, .theme-toggle, .btn {{
                display: none;
            }}
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 1.8em; }}
            .dashboard-summary {{ grid-template-columns: 1fr; }}
            .content {{ padding: 15px; }}
        }}
    </style>
</head>
<body data-csv-file="strategies_{timestamp}.csv">
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="connection-status" id="connectionStatus">
                <span class="status-dot status-live" id="statusDot"></span>
                <span id="statusText">Live Data</span>
            </div>
            <button class="theme-toggle" onclick="toggleTheme()">🌓</button>
            <h1>📊 Advanced Stock Scanner Dashboard</h1>
            <div class="subtitle">Multi-Strategy Trading Signals & Analysis</div>
            <div class="timestamp" id="lastUpdated">Last Updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
        </div>
        
        <!-- Dashboard Summary -->
        <div class="dashboard-summary">
            <div class="summary-card">
                <div class="summary-label">Total Stocks Scanned</div>
                <div class="summary-value">{len(stock_strategies)}</div>
                <div class="summary-subtext">Unique stocks with signals</div>
            </div>
            
            <div class="summary-card">
                <div class="summary-label">Active Strategies</div>
                <div class="summary-value">{len(strategy_summary)}</div>
                <div class="summary-subtext">Strategies with matches</div>
            </div>
            
            <div class="summary-card">
                <div class="summary-label">Market Sentiment</div>
                <div class="summary-value">{sentiment_pct:.0f}%</div>
                <div class="sentiment-badge sentiment-{sentiment.split()[0].lower()}">{sentiment}</div>
            </div>
            
            <div class="summary-card">
                <div class="summary-label">Top Strategy Today</div>
                <div class="summary-value">{top_3_strategies[0]['count'] if top_3_strategies else 0}</div>
                <div class="summary-subtext">{top_3_strategies[0]['name'] if top_3_strategies else 'N/A'}</div>
            </div>
        </div>
        
        <!-- Export Toolbar -->
        <div class="export-toolbar">
            <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search stocks..." onkeyup="filterStocks()">
            <button class="btn refresh-btn" onclick="refreshPrices()">🔄 Refresh Prices</button>
            <button class="btn" onclick="exportToPDF()">📄 Export PDF</button>
            <button class="btn" onclick="copyToClipboard()">📋 Copy to Clipboard</button>
            <button class="btn" onclick="window.print()">🖨️ Print</button>
            <button class="btn" onclick="downloadCSV()">📥 Download CSV</button>
        </div>
        
        <!-- Main Content -->
        <div class="content">
            <div class="section">
                <div class="section-header">
                    <h2 class="section-title">📈 All Stock Signals</h2>
                    <button class="collapse-btn" onclick="toggleSection(this)">▼</button>
                </div>
                
                <div class="collapsible-content">
                    <div class="tabs">
                        <button class="tab active" onclick="switchTab('all-stocks')">All Stocks ({len(stock_strategies)})</button>
"""
    
    # Add strategy tabs
    for strategy in strategy_summary:
        tab_id = strategy['name'].lower().replace(' ', '-')
        html_content += f"""                        <button class="tab" onclick="switchTab('{tab_id}')">{strategy['name']} ({strategy['count']})</button>\n"""
    
    html_content += """                    </div>
                    
                    <!-- All Stocks Table -->
                    <div id="all-stocks" class="tab-content active">
                        <table id="stocksTable">
                            <thead>
                                <tr>
                                    <th>Stock</th>
                                    <th>Action</th>
                                    <th>Price (₹)</th>
                                    <th>Risk</th>
                                    <th>RSI</th>
                                    <th>Vol Ratio</th>
                                    <th>ATR</th>
                                    <th>52W High %</th>
                                    <th>Stop Loss</th>
                                    <th>Target</th>
                                    <th>R:R</th>
                                    <th>Strategy</th>
                                </tr>
                            </thead>
                            <tbody>
"""
    
    # Sort stocks
    action_order = {'BUY': 0, 'SELL': 1, 'HOLD': 2}
    sorted_stocks = sorted(stock_strategies.items(), 
                          key=lambda x: (action_order.get(x[1]['recommendation'], 3), x[0]))
    
    for symbol, data in sorted_stocks:
        action_class = f"action-{data['recommendation'].lower()}"
        vol_class = "volume-highlight" if data['volume_ratio'] > 1.5 else ""
        dist_52w = f"{data['dist_from_52w_high']:+.1f}%" if data['dist_from_52w_high'] != 0 else "N/A"
        
        html_content += f"""                                <tr class="stock-row" data-symbol="{symbol}">
                                    <td><strong>{symbol}</strong></td>
                                    <td><span class="{action_class}">{data['recommendation']}</span></td>
                                    <td>₹{data['close']:.2f}</td>
                                    <td><span class="risk-badge" style="background: {data['risk_color']}; color: white;">{data['risk_level']}</span></td>
                                    <td>{data['rsi_14']:.1f}</td>
                                    <td><span class="{vol_class}">{data['volume_ratio']:.2f}x</span></td>
                                    <td>{data['atr']:.2f}</td>
                                    <td>{dist_52w}</td>
                                    <td>₹{data['stop_loss']:.2f}</td>
                                    <td>₹{data['target_1']:.2f}</td>
                                    <td>{data['rr_ratio_1']:.2f}</td>
                                    <td>{data['strategy']}</td>
                                </tr>
"""
    
    html_content += """                            </tbody>
                        </table>
                    </div>
"""
    
    # Add individual strategy tabs - Generate for ALL strategies
    for strategy in strategy_summary:  # Generate content for all strategies
        tab_id = strategy['name'].lower().replace(' ', '-')
        strategy_stocks = {sym: data for sym, data in stock_strategies.items() 
                          if data['strategy'] == strategy['name']}
        
        html_content += f"""                    <div id="{tab_id}" class="tab-content">
                        <h3>{strategy['name']} Strategy - {strategy['count']} Stocks</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Stock</th>
                                    <th>Price</th>
                                    <th>Risk</th>
                                    <th>R:R</th>
                                    <th>RSI</th>
                                    <th>Vol</th>
                                </tr>
                            </thead>
                            <tbody>
"""
        
        for symbol, data in sorted(strategy_stocks.items(), key=lambda x: x[1]['rr_ratio_1'], reverse=True):
            html_content += f"""                                <tr>
                                    <td><strong>{symbol}</strong></td>
                                    <td>₹{data['close']:.2f}</td>
                                    <td><span class="risk-badge" style="background: {data['risk_color']}; color: white;">{data['risk_level']}</span></td>
                                    <td>{data['rr_ratio_1']:.2f}</td>
                                    <td>{data['rsi_14']:.1f}</td>
                                    <td>{data['volume_ratio']:.2f}x</td>
                                </tr>
"""
        
        html_content += """                            </tbody>
                        </table>
                    </div>
"""
    
    html_content += """                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Dark mode toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        }
        
        // Load saved theme and check connection on page load
        window.addEventListener('DOMContentLoaded', async () => {{
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            
            // Check API connection
            const isConnected = await checkAPIConnection();
            updateConnectionStatus(isConnected);
        });
        
        // Tab switching
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            // Show selected tab content
            const selectedTab = document.getElementById(tabName);
            if (selectedTab) {{
                selectedTab.classList.add('active');
            }}
            // Find and activate the button that was clicked
            const buttons = document.querySelectorAll('.tab');
            buttons.forEach(btn => {{
                if (btn.onclick && btn.onclick.toString().includes(tabName)) {{
                    btn.classList.add('active');
                }}
            }});
        }
        
        // Collapse sections
        function toggleSection(btn) {
            const content = btn.parentElement.nextElementSibling;
            content.classList.toggle('collapsed');
            btn.classList.toggle('collapsed');
        }
        
        // Search filter
        function filterStocks() {
            const input = document.getElementById('searchBox').value.toUpperCase();
            const rows = document.querySelectorAll('.stock-row');
            rows.forEach(row => {
                const symbol = row.getAttribute('data-symbol');
                row.style.display = symbol.includes(input) ? '' : 'none';
            });
        }
        
        // Export to PDF (basic - uses print)
        function exportToPDF() {
            window.print();
        }
        
        // Copy to clipboard
        function copyToClipboard() {{
            const table = document.getElementById('stocksTable');
            let text = '';
            table.querySelectorAll('tr').forEach(row => {{
                const cells = row.querySelectorAll('th, td');
                const rowText = Array.from(cells).map(cell => cell.textContent.trim()).join('\\t');
                text += rowText + '\\n';
            }});
            navigator.clipboard.writeText(text).then(() => {{
                alert('✅ Table copied to clipboard!');
            }});
        }}
        
        // Download CSV
        function downloadCSV() {{
            const csvFile = document.body.getAttribute('data-csv-file');
            window.location.href = csvFile;
        }}
        
        // Check API connection status
        async function checkAPIConnection() {
            try {
                // Try to fetch the CSV file to check if data is accessible
                const csvFile = document.body.getAttribute('data-csv-file');
                const response = await fetch(csvFile, {{ method: 'HEAD' }});
                return response.ok;
            }} catch (error) {{
                return false;
            }}
        }}
        
        // Update connection status indicator
        function updateConnectionStatus(isLive) {{
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            if (isLive) {{
                statusDot.className = 'status-dot status-live';
                statusText.textContent = 'Live Data';
            }} else {{
                statusDot.className = 'status-dot status-cached';
                statusText.textContent = 'Cached Data';
            }}
        }}
        
        // Refresh prices from CSV
        async function refreshPrices() {{
            const refreshBtn = event.target;
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '⏳ Refreshing...';
            
            try {{
                // Check connection first
                const isConnected = await checkAPIConnection();
                updateConnectionStatus(isConnected);
                
                // Get CSV filename from data attribute
                const csvFile = document.body.getAttribute('data-csv-file');
                
                // Parse CSV and update prices
                const response = await fetch(csvFile);
                if (!response.ok) {{
                    throw new Error('Failed to fetch data');
                }}
                
                const csvText = await response.text();
                const lines = csvText.split('\\n');
                const headers = lines[0].split(',');
                
                // Find column indices
                const symbolIdx = headers.indexOf('symbol');
                const closeIdx = headers.indexOf('close');
                
                // Create price map
                const priceMap = {{}};
                for (let i = 1; i < lines.length; i++) {{
                    if (!lines[i].trim()) continue;
                    const values = lines[i].split(',');
                    const symbol = values[symbolIdx]?.trim();
                    const close = parseFloat(values[closeIdx]);
                    if (symbol && !isNaN(close)) {{
                        priceMap[symbol] = close;
                    }}
                }}
                
                // Update all price cells in tables
                let updatedCount = 0;
                document.querySelectorAll('table tbody tr').forEach(row => {{
                    const symbolCell = row.querySelector('td:first-child strong');
                    if (symbolCell) {{
                        const symbol = symbolCell.textContent.trim();
                        if (priceMap[symbol]) {{
                            const priceCell = row.querySelector('td:nth-child(2)');
                            if (priceCell) {{
                                priceCell.textContent = `₹${{priceMap[symbol].toFixed(2)}}`;
                                priceCell.style.backgroundColor = '#d4edda';
                                setTimeout(() => priceCell.style.backgroundColor = '', 1000);
                                updatedCount++;
                            }}
                        }}
                    }}
                }});
                
                // Update timestamp
                const now = new Date();
                const formattedTime = now.toLocaleString('en-US', {{
                    month: 'long',
                    day: 'numeric',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: true
                }});
                document.getElementById('lastUpdated').textContent = `Last Updated: ${{formattedTime}}`;
                
                refreshBtn.innerHTML = '✅ Refreshed!';
                setTimeout(() => {{
                    refreshBtn.innerHTML = '🔄 Refresh Prices';
                    refreshBtn.disabled = false;
                }}, 2000);
                
            }} catch (error) {{
                console.error('Refresh error:', error);
                refreshBtn.innerHTML = '❌ Failed';
                updateConnectionStatus(false);
                setTimeout(() => {{
                    refreshBtn.innerHTML = '🔄 Refresh Prices';
                    refreshBtn.disabled = false;
                }}, 2000);
            }}
        }}
    </script>
</body>
</html>
"""
    
    return html_content


def get_strategy_recommendation(strategy_name):
    """Get recommendation for strategy"""
    bullish_strategies = [
        'Momentum Breakout', 'Trend Following', 'Golden Crossover',
        'Volume Breakout', 'Swing Trading', 'Gap Up', 'Stage 2 Uptrend',
        'Strong Linearity', 'VCP Pattern', 'Pyramiding'
    ]
    
    bearish_strategies = ['Sell Below 10MA', 'Mean Reversion']
    
    if strategy_name in bullish_strategies:
        return 'BUY'
    elif strategy_name in bearish_strategies:
        return 'SELL'
    else:
        return 'HOLD'
