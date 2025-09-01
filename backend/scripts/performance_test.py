#!/usr/bin/env python3
"""
Performance Testing Script for Job Application Helper

Tests GPT-5-mini response times with different reasoning levels:
- minimal, low, medium, high

Measures both streaming and non-streaming performance.
"""

import asyncio
import json
import time
from typing import Dict, List, Any
import httpx
from dataclasses import dataclass
from statistics import mean, median, stdev
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import get_settings
from src.core.llm_providers.factory import get_llm_provider
from src.core.llm_providers.base import ProviderType, GenerationRequest, ContentType


@dataclass
class TestResult:
    """Result of a single performance test."""
    reasoning_level: str
    is_streaming: bool
    response_time: float
    tokens_used: int
    content_length: int
    success: bool
    error: str = None


class PerformanceTester:
    """Performance testing for different reasoning levels."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.settings = get_settings()
        self.test_query = "Write a brief 2-3 sentence response explaining what makes a good software engineer."
        self.reasoning_levels = ["minimal", "low", "medium", "high"]
        self.results: List[TestResult] = []
        
    def test_non_streaming_api(self, reasoning_level: str) -> TestResult:
        """Test non-streaming API performance."""
        start_time = time.time()
        
        try:
            # Prepare request data
            request_data = {
                "message": self.test_query,
                "provider": "openai",
                "model": "gpt-5-mini",
                "reasoning_effort": reasoning_level,
                "session_id": f"perf_test_{reasoning_level}_{int(time.time())}",
                "conversation_history": []
            }
            
            # Make API call
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.api_base_url}/chat/complete",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
            end_time = time.time()
            response_data = response.json()
            
            return TestResult(
                reasoning_level=reasoning_level,
                is_streaming=False,
                response_time=end_time - start_time,
                tokens_used=response_data.get("metadata", {}).get("tokens_used", 0),
                content_length=len(response_data.get("response", "")),
                success=response_data.get("success", False)
            )
            
        except Exception as e:
            return TestResult(
                reasoning_level=reasoning_level,
                is_streaming=False,
                response_time=time.time() - start_time,
                tokens_used=0,
                content_length=0,
                success=False,
                error=str(e)
            )
    
    async def test_streaming_api(self, reasoning_level: str) -> TestResult:
        """Test streaming API performance."""
        start_time = time.time()
        accumulated_content = ""
        chunks_received = 0
        
        try:
            # Prepare request data
            request_data = {
                "message": self.test_query,
                "provider": "openai",
                "model": "gpt-5-mini",
                "reasoning_effort": reasoning_level,
                "session_id": f"perf_test_stream_{reasoning_level}_{int(time.time())}",
                "conversation_history": []
            }
            
            # Make streaming API call
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.api_base_url}/chat/stream",
                    json=request_data,
                    headers={"Accept": "text/event-stream"}
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            
                            if data_str == "[DONE]":
                                break
                                
                            try:
                                data = json.loads(data_str)
                                if data.get("type") == "chunk":
                                    accumulated_content += data.get("content", "")
                                    chunks_received += 1
                                elif data.get("type") == "answer":
                                    accumulated_content += data.get("content", "")
                            except json.JSONDecodeError:
                                continue
            
            end_time = time.time()
            
            return TestResult(
                reasoning_level=reasoning_level,
                is_streaming=True,
                response_time=end_time - start_time,
                tokens_used=0,  # Streaming doesn't provide token count
                content_length=len(accumulated_content),
                success=len(accumulated_content) > 0
            )
            
        except Exception as e:
            return TestResult(
                reasoning_level=reasoning_level,
                is_streaming=True,
                response_time=time.time() - start_time,
                tokens_used=0,
                content_length=0,
                success=False,
                error=str(e)
            )
    
    def test_direct_provider(self, reasoning_level: str) -> TestResult:
        """Test direct provider performance (bypassing API)."""
        start_time = time.time()
        
        try:
            # Get OpenAI provider directly
            provider = get_llm_provider(ProviderType.OPENAI)
            
            # Create generation request
            request = GenerationRequest(
                messages=[{"role": "user", "content": self.test_query}],
                content_type=ContentType.GENERAL_RESPONSE,
                model="gpt-5-mini",
                reasoning_effort=reasoning_level,
                max_tokens=500
            )
            
            # Generate response
            response = provider.generate_content(request)
            
            end_time = time.time()
            
            return TestResult(
                reasoning_level=reasoning_level,
                is_streaming=False,
                response_time=end_time - start_time,
                tokens_used=response.tokens_used,
                content_length=len(response.content),
                success=response.success
            )
            
        except Exception as e:
            return TestResult(
                reasoning_level=reasoning_level,
                is_streaming=False,
                response_time=time.time() - start_time,
                tokens_used=0,
                content_length=0,
                success=False,
                error=str(e)
            )
    
    async def run_comprehensive_test(self, iterations: int = 3):
        """Run comprehensive performance tests."""
        print(f"🚀 Starting performance test with {iterations} iterations per reasoning level")
        print(f"📝 Test query: {self.test_query}")
        print(f"🎯 Reasoning levels: {', '.join(self.reasoning_levels)}")
        print("=" * 80)
        
        # Test 1: Direct Provider (bypassing API)
        print("\n📊 Testing Direct Provider Performance:")
        print("-" * 50)
        
        for level in self.reasoning_levels:
            print(f"\n🔍 Testing reasoning level: {level}")
            level_results = []
            
            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}...", end=" ")
                result = self.test_direct_provider(level)
                level_results.append(result)
                self.results.append(result)
                
                if result.success:
                    print(f"✅ {result.response_time:.2f}s ({result.tokens_used} tokens)")
                else:
                    print(f"❌ Failed: {result.error}")
                
                # Small delay between requests
                await asyncio.sleep(1)
            
            # Calculate statistics
            successful_results = [r for r in level_results if r.success]
            if successful_results:
                times = [r.response_time for r in successful_results]
                tokens = [r.tokens_used for r in successful_results]
                
                print(f"  📈 Statistics:")
                print(f"    Average time: {mean(times):.2f}s (±{stdev(times):.2f}s)")
                print(f"    Median time: {median(times):.2f}s")
                print(f"    Average tokens: {mean(tokens):.0f}")
                print(f"    Success rate: {len(successful_results)}/{len(level_results)}")
        
        # Test 2: Non-streaming API
        print(f"\n\n📊 Testing Non-streaming API Performance:")
        print("-" * 50)
        
        for level in self.reasoning_levels:
            print(f"\n🔍 Testing reasoning level: {level}")
            level_results = []
            
            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}...", end=" ")
                result = self.test_non_streaming_api(level)
                level_results.append(result)
                self.results.append(result)
                
                if result.success:
                    print(f"✅ {result.response_time:.2f}s ({result.tokens_used} tokens)")
                else:
                    print(f"❌ Failed: {result.error}")
                
                # Small delay between requests
                await asyncio.sleep(1)
            
            # Calculate statistics
            successful_results = [r for r in level_results if r.success]
            if successful_results:
                times = [r.response_time for r in successful_results]
                tokens = [r.tokens_used for r in successful_results]
                
                print(f"  📈 Statistics:")
                print(f"    Average time: {mean(times):.2f}s (±{stdev(times):.2f}s)")
                print(f"    Median time: {median(times):.2f}s")
                print(f"    Average tokens: {mean(tokens):.0f}")
                print(f"    Success rate: {len(successful_results)}/{len(level_results)}")
        
        # Test 3: Streaming API
        print(f"\n\n📊 Testing Streaming API Performance:")
        print("-" * 50)
        
        for level in self.reasoning_levels:
            print(f"\n🔍 Testing reasoning level: {level}")
            level_results = []
            
            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}...", end=" ")
                result = await self.test_streaming_api(level)
                level_results.append(result)
                self.results.append(result)
                
                if result.success:
                    print(f"✅ {result.response_time:.2f}s ({result.content_length} chars)")
                else:
                    print(f"❌ Failed: {result.error}")
                
                # Small delay between requests
                await asyncio.sleep(1)
            
            # Calculate statistics
            successful_results = [r for r in level_results if r.success]
            if successful_results:
                times = [r.response_time for r in successful_results]
                chars = [r.content_length for r in successful_results]
                
                print(f"  📈 Statistics:")
                print(f"    Average time: {mean(times):.2f}s (±{stdev(times):.2f}s)")
                print(f"    Median time: {median(times):.2f}s")
                print(f"    Average chars: {mean(chars):.0f}")
                print(f"    Success rate: {len(successful_results)}/{len(level_results)}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "=" * 80)
        print("📋 PERFORMANCE TEST SUMMARY REPORT")
        print("=" * 80)
        
        # Group results by reasoning level and test type
        summary_data = {}
        
        for result in self.results:
            if result.success:
                key = f"{result.reasoning_level}_{'stream' if result.is_streaming else 'direct' if 'direct' in str(result) else 'api'}"
                if key not in summary_data:
                    summary_data[key] = []
                summary_data[key].append(result.response_time)
        
        # Print summary table
        print(f"\n{'Reasoning Level':<12} {'Test Type':<10} {'Avg Time':<10} {'Median':<10} {'Std Dev':<10} {'Samples':<8}")
        print("-" * 70)
        
        for key, times in summary_data.items():
            if times:
                parts = key.split('_')
                reasoning = parts[0]
                test_type = parts[1]
                
                avg_time = mean(times)
                median_time = median(times)
                std_dev = stdev(times) if len(times) > 1 else 0
                
                print(f"{reasoning:<12} {test_type:<10} {avg_time:<10.2f} {median_time:<10.2f} {std_dev:<10.2f} {len(times):<8}")
        
        # Calculate reasoning level impact
        print(f"\n🎯 REASONING LEVEL IMPACT ANALYSIS:")
        print("-" * 50)
        
        for test_type in ['direct', 'api', 'stream']:
            print(f"\n📊 {test_type.upper()} Performance by Reasoning Level:")
            
            level_times = {}
            for key, times in summary_data.items():
                if test_type in key:
                    reasoning = key.split('_')[0]
                    if reasoning not in level_times:
                        level_times[reasoning] = []
                    level_times[reasoning].extend(times)
            
            if level_times:
                # Find baseline (minimal)
                baseline = mean(level_times.get('minimal', [0]))
                
                for level in self.reasoning_levels:
                    if level in level_times:
                        avg_time = mean(level_times[level])
                        if baseline > 0:
                            multiplier = avg_time / baseline
                            print(f"  {level:<8}: {avg_time:.2f}s ({multiplier:.1f}x baseline)")
                        else:
                            print(f"  {level:<8}: {avg_time:.2f}s")
        
        # Overall recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 30)
        
        # Find fastest method
        fastest_method = min(summary_data.items(), key=lambda x: mean(x[1]) if x[1] else float('inf'))
        print(f"• Fastest method: {fastest_method[0]} ({mean(fastest_method[1]):.2f}s average)")
        
        # Find most consistent method
        most_consistent = min(summary_data.items(), key=lambda x: stdev(x[1]) if len(x[1]) > 1 else float('inf'))
        print(f"• Most consistent: {most_consistent[0]} (±{stdev(most_consistent[1]):.2f}s std dev)")
        
        # Reasoning level recommendations
        print(f"• For quick responses: Use 'minimal' reasoning")
        print(f"• For balanced performance: Use 'low' or 'medium' reasoning")
        print(f"• For complex tasks: Use 'high' reasoning (expect 2-3x slower)")


async def main():
    """Main function to run performance tests."""
    print("🔬 GPT-5-mini Performance Testing Tool")
    print("=" * 50)
    
    # Check if API is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8000/health/")
            if response.status_code != 200:
                print("❌ Backend API is not responding properly")
                return
    except Exception as e:
        print(f"❌ Cannot connect to backend API: {e}")
        print("💡 Make sure the backend is running: docker compose up -d")
        return
    
    print("✅ Backend API is running")
    
    # Check OpenAI provider availability
    try:
        from src.core.llm_providers.factory import get_llm_provider
        from src.core.llm_providers.base import ProviderType
        
        provider = get_llm_provider(ProviderType.OPENAI)
        if not provider.is_available():
            print("❌ OpenAI provider is not available")
            print("💡 Make sure OPENAI_API_KEY is set in your environment")
            return
        
        print("✅ OpenAI provider is available")
        
    except Exception as e:
        print(f"❌ Error checking OpenAI provider: {e}")
        return
    
    # Run performance tests
    tester = PerformanceTester()
    
    try:
        await tester.run_comprehensive_test(iterations=3)
        tester.generate_summary_report()
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
