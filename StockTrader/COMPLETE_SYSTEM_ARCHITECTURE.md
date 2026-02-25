# 🏗️ Complete System Architecture with Backtesting

## System Overview
This document shows the complete system architecture including the Scanner (Active Production), Backtesting, and supporting modules.

---

## 📊 Architecture Diagram

```mermaid
graph TB
    subgraph "DATA SOURCES"
        KITE[Kite Connect API<br/>Market Data]
        TELEGRAM[Telegram Bot API<br/>Notifications]
    end
    
    subgraph "BACKTESTING MODULE 🧪"
        BACKTEST_ENGINE[backtesting_engine.py<br/>Core Backtesting Engine]
        VISUALIZER[backtest_visualizer.py<br/>Charts & Reports]
        OPTIMIZER[parameter_optimizer.py<br/>Grid/Random Search]
        STRATEGIES[strategy_examples.py<br/>7 Strategy Templates]
        EXAMPLE[example_backtest.py<br/>Usage Examples]
        
        BACKTEST_ENGINE --> VISUALIZER
        BACKTEST_ENGINE --> OPTIMIZER
        STRATEGIES --> BACKTEST_ENGINE
        EXAMPLE --> BACKTEST_ENGINE
        EXAMPLE --> VISUALIZER
        EXAMPLE --> OPTIMIZER
    end
    
    subgraph "ACTIVE PRODUCTION 🚀"
        AUTO[automated_scanner.py<br/>Main Orchestrator]
        ADVANCED[advanced_scanner.py<br/>Strategy Engine]
        TELEGRAM_NOT[telegram_notifier.py<br/>Telegram Sender]
        HTML_GEN[enhanced_html_generator.py<br/>Dashboard Generator]
        KITE_SCANNER[kite_stock_scanner.py<br/>Base Scanner]
        NIFTY_OI[nifty_oi_tracker.py<br/>Options Tracker]
        
        AUTO --> ADVANCED
        AUTO --> TELEGRAM_NOT
        AUTO --> HTML_GEN
        ADVANCED --> KITE_SCANNER
        NIFTY_OI --> KITE_SCANNER
    end
    
    subgraph "UTILITIES 🔧"
        CREDENTIALS[kite_credentials.txt<br/>API Keys]
        ENV_FILES[.env Files<br/>Environment Config]
        SETUP[setup scripts<br/>Installation Tools]
    end
    
    subgraph "OUTPUTS 📊"
        HTML_DASH[HTML Dashboard<br/>Dark Mode UI]
        TELEGRAM_MSG[Telegram Messages<br/>Real-time Alerts]
        CSV_FILES[CSV Reports<br/>Trade Data]
        CHARTS[Backtest Charts<br/>Performance Viz]
    end
    
    %% Data Flow
    KITE --> KITE_SCANNER
    KITE --> BACKTEST_ENGINE
    
    CREDENTIALS --> KITE_SCANNER
    CREDENTIALS --> BACKTEST_ENGINE
    ENV_FILES --> AUTO
    
    KITE_SCANNER --> ADVANCED
    ADVANCED --> AUTO
    
    AUTO --> HTML_GEN
    AUTO --> TELEGRAM_NOT
    AUTO --> CSV_FILES
    
    HTML_GEN --> HTML_DASH
    TELEGRAM_NOT --> TELEGRAM
    TELEGRAM --> TELEGRAM_MSG
    
    VISUALIZER --> CHARTS
    
    %% Strategy Validation Flow
    STRATEGIES -.->|Validated<br/>Strategies| ADVANCED
    OPTIMIZER -.->|Optimized<br/>Parameters| ADVANCED
    
    style BACKTEST_ENGINE fill:#4a90e2,color:#fff
    style AUTO fill:#e74c3c,color:#fff
    style ADVANCED fill:#e67e22,color:#fff
    style KITE fill:#2ecc71,color:#fff
    style TELEGRAM fill:#3498db,color:#fff
```

---

## 🔄 Complete Workflow Integration

```mermaid
graph LR
    subgraph "STEP 1: Strategy Development 🧪"
        A1[Develop Strategy<br/>in strategy_examples.py]
        A2[Run Backtest<br/>backtesting_engine.py]
        A3[Optimize Parameters<br/>parameter_optimizer.py]
        A4[Walk-Forward<br/>Validation]
        
        A1 --> A2 --> A3 --> A4
    end
    
    subgraph "STEP 2: Integration 🔗"
        B1[Add Strategy to<br/>advanced_scanner.py]
        B2[Configure<br/>Parameters]
        B3[Test on<br/>Live Data]
        
        A4 --> B1 --> B2 --> B3
    end
    
    subgraph "STEP 3: Production 🚀"
        C1[automated_scanner.py<br/>Orchestrates Scan]
        C2[Telegram<br/>Notifications]
        C3[HTML<br/>Dashboard]
        
        B3 --> C1 --> C2
        C1 --> C3
    end
    
    style A4 fill:#2ecc71,color:#fff
    style B3 fill:#f39c12,color:#fff
    style C1 fill:#e74c3c,color:#fff
```

---

## 📁 Module Interaction Map

```mermaid
graph TD
    subgraph "Backtesting Layer"
        BT_ENGINE[Backtesting Engine]
        BT_VIZ[Visualizer]
        BT_OPT[Optimizer]
        BT_STRAT[Strategy Library]
    end
    
    subgraph "Strategy Layer"
        SCANNER[Advanced Scanner]
        STRATEGIES[11+ Trading Strategies]
    end
    
    subgraph "Orchestration Layer"
        AUTO_SCAN[Automated Scanner]
        TELEGRAM[Telegram Notifier]
        HTML[HTML Generator]
    end
    
    subgraph "Data Layer"
        KITE_API[Kite API]
        HISTORICAL[Historical Data]
        REALTIME[Real-time Data]
    end
    
    BT_STRAT -.->|Validated<br/>Strategies| STRATEGIES
    BT_OPT -.->|Optimal<br/>Parameters| STRATEGIES
    
    STRATEGIES --> SCANNER
    
    HISTORICAL --> BT_ENGINE
    REALTIME --> SCANNER
    
    KITE_API --> HISTORICAL
    KITE_API --> REALTIME
    
    SCANNER --> AUTO_SCAN
    AUTO_SCAN --> TELEGRAM
    AUTO_SCAN --> HTML
    
    BT_ENGINE --> BT_VIZ
    BT_ENGINE --> BT_OPT
    BT_STRAT --> BT_ENGINE
```

---

## 🎯 File Classification by Module

### **Backtesting Module** (8 files)
```
Backtesting/
├── 🎯 CORE (4 files)
│   ├── backtesting_engine.py       ⚙️ Main engine
│   ├── backtest_visualizer.py      📊 Charts
│   ├── parameter_optimizer.py      🔍 Optimization
│   └── strategy_examples.py        📈 Strategies
│
├── 📖 EXAMPLES (1 file)
│   └── example_backtest.py         🎓 Tutorial
│
├── 📚 DOCS (2 files)
│   ├── README.md
│   └── QUICKSTART.md
│
└── 📋 CONFIG (1 file)
    └── requirements (2).txt
```

### **Active Production** (6 files)
```
Active_Production/
├── automated_scanner.py            🤖 Main orchestrator
├── advanced_scanner.py             📊 Strategy engine
├── telegram_notifier.py            📱 Notifications
├── enhanced_html_generator.py      🌐 Dashboard
├── kite_stock_scanner.py          🔌 API wrapper
└── nifty_oi_tracker.py            📈 Options tracker
```

### **Utilities** (5 files)
```
Utilities/
├── setup_telegram.py
├── generate_token.py
├── check_nifty_token.py
├── quick_token.py
└── enhanced_html_generator.py
```

### **Archive** (4 files)
```
Archive/
├── closing_momentum_scanner.py
├── eth_swing_screener.py
├── kite_stock_scanner.py
└── Profitable_strategy_scanner.py
```

---

## 🔧 Key Integration Points

### 1. **Strategy Flow**
```
Backtesting/strategy_examples.py
          ↓
  [Backtest & Optimize]
          ↓
Active_Production/advanced_scanner.py
          ↓
  [Live Scanning]
          ↓
Telegram Alerts + HTML Dashboard
```

### 2. **Data Flow**
```
Kite API
    ├─→ Historical Data → Backtesting
    └─→ Real-time Data → Scanner
                              ↓
                        Results/
                              ├─→ HTML Dashboard
                              ├─→ CSV Reports
                              └─→ Telegram Alerts
```

### 3. **Configuration Flow**
```
kite_credentials.txt
        ↓
[Both Modules Use Same Credentials]
        ↓
    Kite API Access
```

---

## 📊 Process Flow Comparison

### **Backtesting Process:**
1. Load historical data (Kite API)
2. Run strategy on historical data
3. Calculate performance metrics
4. Optimize parameters (grid/random search)
5. Validate with walk-forward analysis
6. Generate visualizations
7. Export results

### **Live Scanning Process:**
1. Load real-time data (Kite API)
2. Apply validated strategies
3. Filter signals (7-day window)
4. Deduplicate (latest signal per stock)
5. Send Telegram notifications
6. Generate HTML dashboard
7. Save CSV results

---

## 🎯 Usage Scenarios

### **Scenario 1: New Strategy Development**
```
1. Create strategy in strategy_examples.py
2. Run example_backtest.py
3. Review visualizations
4. Optimize parameters if needed
5. Validate with walk-forward
6. Add to advanced_scanner.py
7. Test live with automated_scanner.py
```

### **Scenario 2: Existing Strategy Optimization**
```
1. Extract strategy from advanced_scanner.py
2. Create backtest version
3. Run parameter_optimizer.py
4. Find optimal parameters
5. Update scanner with new parameters
6. Monitor live performance
```

### **Scenario 3: Performance Analysis**
```
1. Export live trading results (CSV)
2. Load into backtesting_engine.py
3. Generate performance reports
4. Compare with backtest expectations
5. Adjust strategy if needed
```

---

## 🚀 Quick Start Paths

### **For Backtesting:**
```bash
cd Backtesting
python example_backtest.py
```

### **For Live Scanning:**
```bash
cd StockTrader
python Active_Production\automated_scanner.py
```

### **For Strategy Development:**
```bash
# 1. Edit strategy
Backtesting/strategy_examples.py

# 2. Test it
Backtesting/example_backtest.py

# 3. Deploy it
Active_Production/advanced_scanner.py
```

---

*System Integration Complete - Ready for Strategy Development & Live Trading*
