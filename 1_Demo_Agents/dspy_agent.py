"""
Financial Analysis Agent - DSPy Implementation
================================================
This is the DSPy version of the Yahoo Finance ReAct agent.
Based on: https://dspy.ai/tutorials/yahoo_finance_react/

Author: Loc (for Beyond Prompts article)
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file BEFORE other imports
load_dotenv()

# Ensure USER_AGENT is set (suppress warning)
if not os.environ.get('USER_AGENT'):
    os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FinancialAnalysisAgent/1.0'

import dspy
import yfinance as yf
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool


# =============================================================================
# CONFIGURE DSPy
# =============================================================================

def configure_dspy(model_name: str = "openai/gpt-4o-mini"):
    """Configure DSPy with the specified model."""
    lm = dspy.LM(model=model_name)
    dspy.configure(lm=lm)


# =============================================================================
# TOOLS DEFINITION
# =============================================================================

def get_stock_price(ticker: str) -> str:
    """Get current stock price and basic info for a given ticker symbol.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
    
    Returns:
        JSON string with price, change percentage, and company name
    """
    try:
        stock = yf.Ticker(ticker.upper().strip())
        info = stock.info
        hist = stock.history(period="1d")
        
        if hist.empty:
            return f"Could not retrieve data for {ticker}"
        
        current_price = hist['Close'].iloc[-1]
        prev_close = info.get('previousClose', current_price)
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        
        result = {
            "ticker": ticker.upper(),
            "price": round(current_price, 2),
            "change_percent": round(change_pct, 2),
            "company": info.get('longName', ticker),
            "market_cap": info.get('marketCap', 'N/A'),
            "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52_week_low": info.get('fiftyTwoWeekLow', 'N/A'),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"


def compare_stocks(tickers: str) -> str:
    """Compare multiple stocks side by side.

    Args:
        tickers: Comma-separated list of ticker symbols (e.g., 'AAPL,GOOGL,MSFT')

    Returns:
        JSON string with comparison data for all tickers
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        comparison = []

        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

                comparison.append({
                    "ticker": ticker,
                    "company": info.get('longName', ticker),
                    "price": round(current_price, 2),
                    "change_percent": round(change_pct, 2),
                    "pe_ratio": info.get('trailingPE', 'N/A'),
                    "market_cap": info.get('marketCap', 'N/A'),
                })

        return json.dumps(comparison, indent=2)
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"


def get_yahoo_finance_news(ticker: str) -> str:
    """Get latest financial news for a given ticker from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')

    Returns:
        JSON string with recent news articles about the stock
    """
    try:
        import asyncio

        # Create the LangChain tool
        yahoo_tool = YahooFinanceNewsTool()

        # Run the async function synchronously
        try:
            # Try to get the running event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, yahoo_tool._arun(ticker))
                    result = future.result()
            else:
                # If no loop is running, we can use asyncio.run
                result = asyncio.run(yahoo_tool._arun(ticker))
        except RuntimeError:
            # Fallback: create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(yahoo_tool._arun(ticker))
            finally:
                loop.close()

        return result
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


# =============================================================================
# DSPy FINANCIAL ANALYSIS AGENT
# =============================================================================

class FinancialAnalysisAgentDSPy(dspy.Module):
    """
    ReAct agent for financial analysis using Yahoo Finance data.

    This is the DSPy implementation for comparison with LangGraph.
    """

    def __init__(self, max_iters: int = 6):
        super().__init__()

        # Combine all tools (using synchronous wrapper for Yahoo Finance News)
        self.tools = [
            get_yahoo_finance_news,  # Yahoo Finance News (synchronous wrapper)
            get_stock_price,
            compare_stocks
        ]

        # Initialize ReAct with same max_iters as LangGraph version
        self.react = dspy.ReAct(
            signature="financial_query -> analysis_response",
            tools=self.tools,
            max_iters=max_iters
        )

        self.max_iters = max_iters

    def forward(self, financial_query: str):
        """Run the ReAct agent on a financial query."""
        return self.react(financial_query=financial_query)


# =============================================================================
# WRAPPER CLASS (unified interface for benchmarking)
# =============================================================================

class FinancialAnalysisAgentDSPyWrapper:
    """
    Wrapper to provide consistent interface with LangGraph version.
    """
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini", max_iters: int = 6):
        configure_dspy(model_name)
        self.agent = FinancialAnalysisAgentDSPy(max_iters=max_iters)
        self.model_name = model_name
    
    def __call__(self, financial_query: str):
        """Run the agent and return response."""
        result = self.agent(financial_query=financial_query)
        return result


# =============================================================================
# DEMO FUNCTION
# =============================================================================

def run_financial_demo():
    """Demo of the DSPy financial analysis agent."""
    
    print("=" * 60)
    print("DSPy Financial Analysis Agent Demo")
    print("=" * 60)
    
    # Initialize agent
    agent = FinancialAnalysisAgentDSPyWrapper(model_name="openai/gpt-4o-mini")
    
    # Same example queries as LangGraph version
    queries = [
        "What's the latest news about Apple (AAPL) and how might it affect the stock price?",
        "Compare AAPL, GOOGL, and MSFT performance",
        "Find recent Tesla news and analyze sentiment"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("=" * 60)
        
        start_time = datetime.now()
        response = agent(financial_query=query)
        end_time = datetime.now()
        
        print(f"\nAnalysis:\n{response.analysis_response}")
        print(f"\n[Time: {(end_time - start_time).total_seconds():.2f}s]")
        print("-" * 60)


if __name__ == "__main__":
    run_financial_demo()