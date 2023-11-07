import sys
import os
import argparse
import timeit
from sympy import symbols, expand

def run_benchmark(n):
    a_variables = [symbols(f'a{i}') for i in range(n)]
    e = sum(a_variables)
    f = -sum(a_variables[1:])

    def benchmark():
        e_squared = e**2
        e_substituted = e_squared.subs(a_variables[0], f)
        expanded_result = expand(e_substituted)

    time_taken = timeit.timeit(benchmark, number=1000)  # Measure time over 1000 iterations
    print(f"Time taken for n={n}: {time_taken:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=100, help='Number of variables (default: 100)')
    args = parser.parse_args()
    
    run_benchmark(args.n)
