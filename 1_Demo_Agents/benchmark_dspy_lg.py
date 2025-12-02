"""
DSPy vs LangGraph Benchmark
============================
Compare performance, token usage, latency, and output quality
between DSPy and LangGraph implementations of the Financial Analysis Agent.

Author: Loc (for Beyond Prompts article)
"""

import json
import time
import statistics
from datetime import datetime
from typing import Callable
from dataclasses import dataclass, field, asdict
import traceback

from dotenv import load_dotenv

# Load environment variables from .env file BEFORE other imports
load_dotenv()

# We'll import the agents conditionally based on what's installed
try:
    from financial_agent_dspy import FinancialAnalysisAgentDSPyWrapper
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("Warning: DSPy agent not available")

try:
    from financial_agent_langgraph import FinancialAnalysisAgentLangGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph agent not available")


# =============================================================================
# BENCHMARK DATA STRUCTURES
# =============================================================================

@dataclass
class SingleRunResult:
    """Result of a single query run."""
    query: str
    framework: str
    success: bool
    response: str = ""
    latency_seconds: float = 0.0
    iterations: int = 0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a framework."""
    framework: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    avg_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    std_latency: float = 0.0
    avg_response_length: int = 0
    avg_iterations: float = 0.0


# =============================================================================
# BENCHMARK QUERIES
# =============================================================================

BENCHMARK_QUERIES = [
    # Simple queries
    {
        "query": "What is the current stock price of Apple (AAPL)?",
        "category": "simple",
        "expected_tools": ["get_stock_price"]
    },
    {
        "query": "Get the latest news about Microsoft",
        "category": "simple",
        "expected_tools": ["get_yahoo_finance_news"]
    },
    
    # Medium complexity
    {
        "query": "What's the latest news about Apple (AAPL) and how might it affect the stock price?",
        "category": "medium",
        "expected_tools": ["get_yahoo_finance_news", "get_stock_price"]
    },
    {
        "query": "Compare AAPL, GOOGL, and MSFT performance",
        "category": "medium",
        "expected_tools": ["compare_stocks"]
    },
    
    # Complex queries
    {
        "query": "Find recent Tesla news and analyze the sentiment. Also get the current stock price and tell me if it's a good time to buy.",
        "category": "complex",
        "expected_tools": ["get_yahoo_finance_news", "get_stock_price"]
    },
    {
        "query": "I'm considering investing in tech stocks. Compare Apple, Google, Microsoft, and Amazon. Also get any recent news that might affect these stocks.",
        "category": "complex",
        "expected_tools": ["compare_stocks", "get_yahoo_finance_news"]
    },
]


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class FrameworkBenchmark:
    """Run benchmarks on a single framework."""
    
    def __init__(self, name: str, agent_factory: Callable, model_name: str = "gpt-4o-mini"):
        self.name = name
        self.agent = agent_factory(model_name=model_name)
        self.results: list[SingleRunResult] = []
    
    def run_single_query(self, query: str) -> SingleRunResult:
        """Run a single query and measure performance."""
        result = SingleRunResult(
            query=query,
            framework=self.name,
            success=False
        )
        
        try:
            start_time = time.perf_counter()
            response = self.agent(financial_query=query)
            end_time = time.perf_counter()
            
            result.success = True
            result.latency_seconds = end_time - start_time
            result.response = response.analysis_response
            
            # Get iterations if available (LangGraph tracks this)
            if hasattr(response, 'iterations'):
                result.iterations = response.iterations
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.latency_seconds = 0.0
            print(f"  Error: {e}")
            traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def run_benchmark(self, queries: list[dict], runs_per_query: int = 1) -> list[SingleRunResult]:
        """Run full benchmark suite."""
        print(f"\n{'='*60}")
        print(f"Running {self.name} Benchmark")
        print(f"{'='*60}")
        
        for q_data in queries:
            query = q_data["query"]
            category = q_data["category"]
            
            for run in range(runs_per_query):
                print(f"\n[{category.upper()}] Query: {query[:50]}...")
                print(f"  Run {run + 1}/{runs_per_query}")
                
                result = self.run_single_query(query)
                
                if result.success:
                    print(f"  ✓ Success - {result.latency_seconds:.2f}s")
                    print(f"  Response length: {len(result.response)} chars")
                else:
                    print(f"  ✗ Failed - {result.error}")
        
        return self.results
    
    def get_metrics(self) -> BenchmarkMetrics:
        """Calculate aggregate metrics."""
        successful = [r for r in self.results if r.success]
        
        metrics = BenchmarkMetrics(framework=self.name)
        metrics.total_runs = len(self.results)
        metrics.successful_runs = len(successful)
        metrics.failed_runs = metrics.total_runs - metrics.successful_runs
        
        if successful:
            latencies = [r.latency_seconds for r in successful]
            metrics.avg_latency = statistics.mean(latencies)
            metrics.min_latency = min(latencies)
            metrics.max_latency = max(latencies)
            metrics.std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            metrics.avg_response_length = int(statistics.mean([len(r.response) for r in successful]))
            
            iterations = [r.iterations for r in successful if r.iterations > 0]
            if iterations:
                metrics.avg_iterations = statistics.mean(iterations)
        
        return metrics


# =============================================================================
# COMPARISON REPORT
# =============================================================================

def generate_comparison_report(
    dspy_metrics: BenchmarkMetrics | None,
    langgraph_metrics: BenchmarkMetrics | None,
    dspy_results: list[SingleRunResult] | None,
    langgraph_results: list[SingleRunResult] | None
) -> str:
    """Generate a markdown comparison report."""
    
    report = []
    report.append("# DSPy vs LangGraph: Financial Agent Benchmark Results\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    # Summary Table
    report.append("## Summary Metrics\n")
    report.append("| Metric | DSPy | LangGraph | Winner |")
    report.append("|--------|------|-----------|--------|")
    
    if dspy_metrics and langgraph_metrics:
        # Success Rate
        dspy_sr = dspy_metrics.successful_runs / dspy_metrics.total_runs * 100 if dspy_metrics.total_runs > 0 else 0
        lg_sr = langgraph_metrics.successful_runs / langgraph_metrics.total_runs * 100 if langgraph_metrics.total_runs > 0 else 0
        winner = "DSPy" if dspy_sr > lg_sr else ("LangGraph" if lg_sr > dspy_sr else "Tie")
        report.append(f"| Success Rate | {dspy_sr:.1f}% | {lg_sr:.1f}% | {winner} |")
        
        # Average Latency
        winner = "DSPy" if dspy_metrics.avg_latency < langgraph_metrics.avg_latency else "LangGraph"
        report.append(f"| Avg Latency | {dspy_metrics.avg_latency:.2f}s | {langgraph_metrics.avg_latency:.2f}s | {winner} |")
        
        # Min Latency
        winner = "DSPy" if dspy_metrics.min_latency < langgraph_metrics.min_latency else "LangGraph"
        report.append(f"| Min Latency | {dspy_metrics.min_latency:.2f}s | {langgraph_metrics.min_latency:.2f}s | {winner} |")
        
        # Max Latency
        winner = "DSPy" if dspy_metrics.max_latency < langgraph_metrics.max_latency else "LangGraph"
        report.append(f"| Max Latency | {dspy_metrics.max_latency:.2f}s | {langgraph_metrics.max_latency:.2f}s | {winner} |")
        
        # Response Length (more detail = better?)
        winner = "DSPy" if dspy_metrics.avg_response_length > langgraph_metrics.avg_response_length else "LangGraph"
        report.append(f"| Avg Response Length | {dspy_metrics.avg_response_length} chars | {langgraph_metrics.avg_response_length} chars | {winner} |")
    
    report.append("\n")
    
    # Detailed Results
    report.append("## Detailed Results by Query\n")
    
    if dspy_results and langgraph_results:
        # Group by query
        dspy_by_query = {r.query: r for r in dspy_results}
        lg_by_query = {r.query: r for r in langgraph_results}
        
        for query in dspy_by_query.keys():
            dspy_r = dspy_by_query.get(query)
            lg_r = lg_by_query.get(query)
            
            report.append(f"### Query: {query[:60]}...\n")
            report.append("| Metric | DSPy | LangGraph |")
            report.append("|--------|------|-----------|")
            
            if dspy_r and lg_r:
                report.append(f"| Status | {'✓' if dspy_r.success else '✗'} | {'✓' if lg_r.success else '✗'} |")
                report.append(f"| Latency | {dspy_r.latency_seconds:.2f}s | {lg_r.latency_seconds:.2f}s |")
                report.append(f"| Response Length | {len(dspy_r.response)} | {len(lg_r.response)} |")
            
            report.append("\n")
    
    # Analysis
    report.append("## Analysis\n")
    report.append("### Key Observations\n")
    
    if dspy_metrics and langgraph_metrics:
        # Latency comparison
        latency_diff = abs(dspy_metrics.avg_latency - langgraph_metrics.avg_latency)
        faster = "DSPy" if dspy_metrics.avg_latency < langgraph_metrics.avg_latency else "LangGraph"
        pct_diff = latency_diff / max(dspy_metrics.avg_latency, langgraph_metrics.avg_latency) * 100
        
        report.append(f"1. **Latency**: {faster} is {pct_diff:.1f}% faster on average\n")
        
        # Reliability
        if dspy_metrics.successful_runs == dspy_metrics.total_runs and langgraph_metrics.successful_runs == langgraph_metrics.total_runs:
            report.append("2. **Reliability**: Both frameworks achieved 100% success rate\n")
        else:
            more_reliable = "DSPy" if dspy_metrics.successful_runs > langgraph_metrics.successful_runs else "LangGraph"
            report.append(f"2. **Reliability**: {more_reliable} had higher success rate\n")
    
    report.append("\n### Recommendations\n")
    report.append("- **Choose DSPy** if: You need prompt optimization, want declarative pipelines, or plan to swap models frequently\n")
    report.append("- **Choose LangGraph** if: You need stateful workflows, complex branching logic, or enterprise features like audit trails\n")
    
    return "\n".join(report)


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_full_benchmark(
    model_name: str = "gpt-4o-mini",
    runs_per_query: int = 1,
    output_file: str = "benchmark_results.md"
):
    """Run the complete benchmark comparison."""
    
    print("=" * 70)
    print("DSPy vs LangGraph: Financial Agent Benchmark")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Queries: {len(BENCHMARK_QUERIES)}")
    print(f"Runs per query: {runs_per_query}")
    print("=" * 70)
    
    dspy_metrics = None
    langgraph_metrics = None
    dspy_results = None
    langgraph_results = None
    
    # Run DSPy benchmark
    if DSPY_AVAILABLE:
        try:
            dspy_bench = FrameworkBenchmark(
                name="DSPy",
                agent_factory=lambda model_name: FinancialAnalysisAgentDSPyWrapper(model_name=f"openai/{model_name}"),
                model_name=model_name
            )
            dspy_results = dspy_bench.run_benchmark(BENCHMARK_QUERIES, runs_per_query)
            dspy_metrics = dspy_bench.get_metrics()
        except Exception as e:
            print(f"DSPy benchmark failed: {e}")
            traceback.print_exc()
    else:
        print("Skipping DSPy benchmark (not installed)")
    
    # Run LangGraph benchmark
    if LANGGRAPH_AVAILABLE:
        try:
            langgraph_bench = FrameworkBenchmark(
                name="LangGraph",
                agent_factory=FinancialAnalysisAgentLangGraph,
                model_name=model_name
            )
            langgraph_results = langgraph_bench.run_benchmark(BENCHMARK_QUERIES, runs_per_query)
            langgraph_metrics = langgraph_bench.get_metrics()
        except Exception as e:
            print(f"LangGraph benchmark failed: {e}")
            traceback.print_exc()
    else:
        print("Skipping LangGraph benchmark (not installed)")
    
    # Generate report
    report = generate_comparison_report(
        dspy_metrics, langgraph_metrics,
        dspy_results, langgraph_results
    )
    
    # Save report
    with open(output_file, "w") as f:
        f.write(report)
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Report saved to: {output_file}")
    
    # Print summary
    if dspy_metrics:
        print(f"\nDSPy: {dspy_metrics.successful_runs}/{dspy_metrics.total_runs} success, avg {dspy_metrics.avg_latency:.2f}s")
    if langgraph_metrics:
        print(f"LangGraph: {langgraph_metrics.successful_runs}/{langgraph_metrics.total_runs} success, avg {langgraph_metrics.avg_latency:.2f}s")
    
    # Return results for programmatic use
    return {
        "dspy": {"metrics": asdict(dspy_metrics) if dspy_metrics else None, "results": dspy_results},
        "langgraph": {"metrics": asdict(langgraph_metrics) if langgraph_metrics else None, "results": langgraph_results}
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark DSPy vs LangGraph")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--runs", type=int, default=1, help="Runs per query")
    parser.add_argument("--output", default="benchmark_results.md", help="Output file")
    
    args = parser.parse_args()
    
    run_full_benchmark(
        model_name=args.model,
        runs_per_query=args.runs,
        output_file=args.output
    )