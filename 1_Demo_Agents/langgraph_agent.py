"""
Financial Analysis Agent - LangGraph Implementation
=====================================================
This is a LangGraph equivalent of the DSPy Yahoo Finance ReAct agent.
Built for comparing DSPy vs LangGraph performance.

Author: Loc (for Beyond Prompts article)
"""

import json
import os
import operator
from typing import Annotated, Sequence, TypedDict, Literal
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file BEFORE other imports
load_dotenv()

# Ensure USER_AGENT is set (suppress warning)
if not os.environ.get('USER_AGENT'):
    os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FinancialAnalysisAgent/1.0'

import yfinance as yf
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """State that flows through the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Track iterations for comparison with DSPy's max_iters=6
    iteration_count: int


# =============================================================================
# TOOLS DEFINITION
# =============================================================================

@tool
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


@tool
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


@tool
def get_yahoo_finance_news(query: str) -> str:
    """Get the latest financial news from Yahoo Finance for a given query.
    
    Args:
        query: Search query - typically a company name or ticker symbol
    
    Returns:
        Recent news articles related to the query
    """
    try:
        yahoo_tool = YahooFinanceNewsTool()
        result = yahoo_tool.run(query)
        return result
    except Exception as e:
        return f"Error fetching news: {str(e)}"


# Collect all tools
TOOLS = [get_stock_price, compare_stocks, get_yahoo_finance_news]


# =============================================================================
# AGENT NODE (LLM with tool calling)
# =============================================================================

def create_agent_node(model: ChatOpenAI, tools: list[BaseTool]):
    """Create the agent node that decides what to do next."""
    
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)
    
    # System prompt for financial analysis
    system_prompt = """You are an expert financial analyst assistant. Your job is to:

1. Analyze financial queries using available tools
2. Fetch real-time stock prices and news
3. Provide clear, actionable investment insights
4. Compare stocks when asked

Always use the tools to get real data before making any analysis.
Be concise but thorough in your analysis.
When analyzing news, consider both positive and negative sentiment.

Available tools:
- get_stock_price: Get current price for a single stock
- compare_stocks: Compare multiple stocks (comma-separated tickers)
- get_yahoo_finance_news: Get latest news for a company/ticker
"""
    
    def agent(state: AgentState) -> dict:
        """The agent node - calls LLM to decide next action."""
        messages = state["messages"]
        
        # Add system prompt if this is the first call
        if len(messages) == 1:
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        
        # Call the model
        response = model_with_tools.invoke(messages)
        
        # Update iteration count
        new_count = state.get("iteration_count", 0) + 1
        
        return {
            "messages": [response],
            "iteration_count": new_count
        }
    
    return agent


# =============================================================================
# ROUTING LOGIC
# =============================================================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to continue with tools or end the conversation."""
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)
    
    # Safety: max iterations (matching DSPy's max_iters=6)
    if iteration_count >= 6:
        return "end"
    
    # If the LLM made tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end
    return "end"


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

def build_financial_agent(model_name: str = "gpt-4o-mini") -> StateGraph:
    """
    Build the LangGraph financial analysis agent.
    
    This mirrors the DSPy ReAct agent structure:
    - ReAct loop with max 6 iterations
    - Same tools (stock price, comparison, news)
    - Same reasoning pattern
    """
    
    # Initialize the LLM
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", create_agent_node(llm, TOOLS))
    workflow.add_node("tools", ToolNode(TOOLS))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges (the ReAct loop)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tools always go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()


# =============================================================================
# WRAPPER CLASS (mirrors DSPy's FinancialAnalysisAgent interface)
# =============================================================================

class FinancialAnalysisAgentLangGraph:
    """
    LangGraph Financial Analysis Agent.
    
    Interface matches DSPy version for fair comparison:
    - agent(financial_query=query) -> response with analysis_response attribute
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.graph = build_financial_agent(model_name)
        self.model_name = model_name
    
    def __call__(self, financial_query: str) -> "AnalysisResponse":
        """Run the agent on a financial query."""
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=financial_query)],
            "iteration_count": 0
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract final response
        final_message = result["messages"][-1]
        
        # Return in DSPy-compatible format
        return AnalysisResponse(
            analysis_response=final_message.content,
            iterations=result["iteration_count"],
            messages=result["messages"]
        )


class AnalysisResponse:
    """Response object matching DSPy's output format."""
    
    def __init__(self, analysis_response: str, iterations: int, messages: list):
        self.analysis_response = analysis_response
        self.iterations = iterations
        self.messages = messages
    
    def __repr__(self):
        return f"AnalysisResponse(iterations={self.iterations}, response_length={len(self.analysis_response)})"


# =============================================================================
# DEMO FUNCTION
# =============================================================================

def run_financial_demo():
    """Demo of the LangGraph financial analysis agent."""
    
    print("=" * 60)
    print("LangGraph Financial Analysis Agent Demo")
    print("=" * 60)
    
    # Initialize agent
    agent = FinancialAnalysisAgentLangGraph(model_name="gpt-4o-mini")
    
    # Same example queries as DSPy tutorial
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
        print(f"\n[Iterations: {response.iterations}, Time: {(end_time - start_time).total_seconds():.2f}s]")
        print("-" * 60)


if __name__ == "__main__":
    run_financial_demo()