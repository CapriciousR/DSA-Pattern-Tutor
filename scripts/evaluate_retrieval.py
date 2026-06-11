import os
import sys
import time

# Add the root directory to the python path so it can find 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import HybridRouterRetriever

# Comprehensive 22-Case DSA Test Suite
TEST_CASES = [
    {"id": 1, "query": "Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order. The same number may be chosen from candidates an unlimited number of times.", "expected_sources": ["combination_sum.py", "combination_sum_i.py"]},
    {"id": 2, "query": "Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target. Each number in candidates may only be used once in the combination. Note: The solution set must not contain duplicate combinations.", "expected_sources": ["combination_sum_ii.py", "combination_sum_2.py"]},
    {"id": 3, "query": "Given an integer array nums of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets. Return the solution in any order.", "expected_sources": ["subsets.py", "power_set.py", "subset_generation.py"]},
    {"id": 4, "query": "Given an integer array nums that may contain duplicates, return all possible subsets (the power set). The solution set must not contain duplicate subsets. Return the solution in any order.", "expected_sources": ["subsets_ii.py", "subsets_2.py"]},
    {"id": 5, "query": "Given an integer n, generate all combinations of well-formed parentheses.", "expected_sources": ["generate_parentheses.py", "valid_parentheses_generation.py"]},
    {"id": 6, "query": "Find the shortest path from a starting node to all other nodes in a weighted graph where all edge weights are non-negative.", "expected_sources": ["dijkstra.py", "dijkstras_algorithm.py"]},
    {"id": 7, "query": "Find the shortest paths between all pairs of vertices in a directed weighted graph. The graph may contain negative edge weights, but no negative cycles.", "expected_sources": ["floyd_warshall.py", "graphs_floyd_warshall.py", "all_pairs_shortest_path.py"]},
    {"id": 8, "query": "Find the shortest path from a single source vertex to all other vertices in a weighted digraph. Unlike other algorithms, this can handle graphs containing negative weight edges, but will flag if a negative cycle is present.", "expected_sources": ["bellman_ford.py", "bellman_ford_algorithm.py"]},
    {"id": 9, "query": "Given a directed graph, find all pairs shortest paths using a combination of Dijkstra and Bellman-Ford, optimizing for sparse graphs.", "expected_sources": ["johnson.py", "johnsons_algorithm.py"]},
    {"id": 10, "query": "Given the root of a binary tree, return its maximum depth. The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.", "expected_sources": ["max_depth_binary_tree.py", "depth_first_search.py", "binary_tree_depth.py"]},
    {"id": 11, "query": "Given the root of a binary tree, return the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.", "expected_sources": ["diameter_of_binary_tree.py", "tree_diameter.py"]},
    {"id": 12, "query": "Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).", "expected_sources": ["binary_tree_level_order_traversal.py", "bfs_binary_tree.py", "level_order_traversal.py"]},
    {"id": 13, "query": "Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index.", "expected_sources": ["binary_search.py"]},
    {"id": 14, "query": "Given an array of integers nums, sort the array in ascending order using a stable divide-and-conquer algorithm with O(n log n) time complexity.", "expected_sources": ["merge_sort.py"]},
    {"id": 15, "query": "Sort an array of integers in place using a partitioning scheme where an element is chosen as a pivot and elements are rearranged around it.", "expected_sources": ["quick_sort.py", "inplace_quicksort.py"]},
    {"id": 16, "query": "Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array. Optimize for O(n) time complexity and O(1) auxiliary space using an in-place cyclic pointer placement.", "expected_sources": ["cyclic_sort.py", "missing_number.py"]},
    {"id": 17, "query": "Given a set of items, each with a weight and a value, determine the items to include in a collection so that the total weight is less than or equal to a given limit and the total value is maximized.", "expected_sources": ["knapsack.py", "zero_one_knapsack.py", "recursive_approach_knapsack.py"]},
    {"id": 18, "query": "Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.", "expected_sources": ["longest_common_subsequence.py", "lcs.py", "distinct_subsequences.py"]},
    {"id": 19, "query": "You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?", "expected_sources": ["climbing_stairs.py", "fibonacci.py"]},
    {"id": 20, "query": "Implement an LRU (Least Recently Used) cache class which supports get and put operations in O(1) time complexity.", "expected_sources": ["lru_cache.py", "least_recently_used_cache.py"]},
    {"id": 21, "query": "Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, find the area of the largest rectangle in the histogram.", "expected_sources": ["largest_rectangle_in_histogram.py", "monotonic_stack_histogram.py", "largest_rectangle_histogram.py"]},
    {"id": 22, "query": "Given an array of integers string representing data, implement most significant digit radix sort to sort the strings lexicographically.", "expected_sources": ["msd_radix_sort.py", "radix_sort.py"]}
]

def run_evaluation():
    print("\nInitializing Production Metadata Router Engine...\n")
    retriever = HybridRouterRetriever()
    
    top_1_hits = 0
    top_3_hits = 0
    total = len(TEST_CASES)
    total_time = 0
    
    for test in TEST_CASES:
        query = test["query"]
        expected_list = test["expected_sources"]
        
        start_time = time.time()
        
        # 1. We manually call route_query just to print it for the benchmark logs
        routed_category = retriever.route_query(query)
        
        # 2. Invoke the full pipeline (Routes + Retrieves)
        retrieved_docs = retriever.invoke(query, k=3)
        
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed
        
        retrieved_sources = [
            doc.metadata.get("source", "").split("\\")[-1].split("/")[-1] 
            for doc in retrieved_docs
        ]
        
        print(f"\nTest {test['id']}: {query[:60]}...")
        print(f"  🏷️  Routed to: {routed_category} (in {elapsed:.2f}s)")
        print(f"  🎯 Expected: {expected_list}")
        print(f"  📦 Retrieved: {retrieved_sources}")
        
        is_top_1 = False
        is_top_3 = False
        
        if len(retrieved_sources) > 0 and retrieved_sources[0] in expected_list:
            is_top_1 = True
            is_top_3 = True
        else:
            for src in retrieved_sources:
                if src in expected_list:
                    is_top_3 = True
                    break
                    
        if is_top_1:
            top_1_hits += 1
            top_3_hits += 1
            print("  ✅ Perfect Match (Top-1)")
        elif is_top_3:
            top_3_hits += 1
            print("  ⚠️ Soft Match (Top-3)")
        else:
            print("  ❌ Severe Miss")
        print("-" * 60)
        
    print("\n📊 PHASE 6 BENCHMARK SUMMARY METRICS (Metadata Router + Hybrid):")
    print(f"Top-1 Categorical Accuracy: {(top_1_hits / total) * 100:.1f}%")
    print(f"Top-3 Categorical Accuracy: {(top_3_hits / total) * 100:.1f}%")
    print(f"Average Pipeline Time: {total_time / total:.2f}s per query")

if __name__ == "__main__":
    run_evaluation()