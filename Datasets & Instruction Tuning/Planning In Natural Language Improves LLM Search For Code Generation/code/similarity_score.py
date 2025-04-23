from difflib import SequenceMatcher

def calculate_similarity(solutions):
    if len(solutions) <= 1:
        return 0.0
    
    total_similarity = 0.0
    comparisons = 0
    
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            similarity = SequenceMatcher(None, solutions[i], solutions[j]).ratio()
            total_similarity += similarity
            comparisons += 1
            print(f"  Similarity between solution {i+1} and {j+1}: {similarity:.4f}")
    
    return total_similarity / comparisons if comparisons > 0 else 0.0

def analyze_similarity(problem_name, sketch_solutions, baseline_solutions):
    print(f"\n{'-' * 60}")
    print(f"Problem: {problem_name}")
    print(f"{'-' * 60}")
    
    print("\nSketch-based solutions:")
    sketch_similarity = calculate_similarity(sketch_solutions)
    
    print("\nBaseline solutions:")
    baseline_similarity = calculate_similarity(baseline_solutions)
    
    print(f"\nResults for {problem_name}:")
    print(f"  Sketch-based solutions average similarity: {sketch_similarity:.4f}")
    print(f"  Baseline solutions average similarity: {baseline_similarity:.4f}")
    print(f"  Difference (sketch - baseline): {sketch_similarity - baseline_similarity:.4f}")
    
    return {
        'sketch_similarity': sketch_similarity,
        'baseline_similarity': baseline_similarity
    }

def main():
    results = {}
    
    # Row Max Score Solutions
    row_max_sketch_solutions = [
        """
import heapq

def calculate_final_score(nums):
    # Initialize the score to 0
    score = 0
    
    # Continue the process until the matrix becomes empty
    while nums:
        max_values = []  # Store the maximum values from each row
        
        # Find the maximum value from each row
        for row in nums:
            max_val = max(row)
            max_values.append(max_val)
            row.remove(max_val)  # Remove the maximum value from the row
        
        # Get the overall maximum value from the max_values list
        max_overall = max(max_values)
        
        # Update the score by adding the overall maximum value
        score += max_overall
        
        # Remove rows that are now empty after selecting the maximum values
        nums = [row for row in nums if row]
    
    return score
        """,
        """
import heapq

def final_score(nums):
    if not nums:
        return 0  # If matrix is empty, return 0 as the final score
    
    score = 0
    while True:
        max_values = []  # List to store the maximum value in each row
        for row in nums:
            if row:
                max_val = max(row)
                max_values.append(max_val)
                row.remove(max_val)  # Remove the maximum value from the row
            else:
                max_values.append(float('-inf'))  # If row is empty, consider it as negative infinity
        
        if max(max_values) == float('-inf'):
            break  # If all rows are empty, exit the loop
        
        max_score = max(max_values)  # Get the maximum value among all removed values
        score += max_score  # Add the maximum value to the score
    
    return score
        """,
        """
import heapq

def final_score(nums):
    if not nums:  # Handle edge case when matrix is initially empty
        return 0
    
    score = 0
    heap = []  # Initialize an empty heap
    
    # Push the largest element from each row into the heap
    for row in nums:
        heapq.heappush(heap, -max(row))  # Negate the value for max heap
    
    while heap:
        max_val = -heapq.heappop(heap)  # Get the largest removed value
        score += max_val  # Add the largest value to the score
        
        # Update the matrix by removing the selected max value from each row
        for row in nums:
            if max_val in row:
                row.remove(max_val)
        
        # Rebuild the heap with the new largest values from each row
        heap = []
        for row in nums:
            if row:
                heapq.heappush(heap, -max(row))
    
    return score
        """
    ]
    
    row_max_baseline_solutions = [
        """
def max_score(nums):
    # Initialize the total score
    score = 0
    
    # Iterate until the matrix becomes empty
    while nums:
        max_values = []  # Stores the maximum values from each row
        
        # Find the maximum value in each row and remove it
        for row in nums:
            max_val = max(row)
            row.remove(max_val)
            max_values.append(max_val)
        
        # Find the maximum value among all the removed values
        max_removed = max(max_values)
        
        # Add the maximum removed value to the total score
        score += max_removed
        
        # Remove all instances of the maximum removed value from the matrix
        for row in nums:
            row[:] = [val for val in row if val != max_removed]
        
        # Remove empty rows from the matrix
        nums = [row for row in nums if row]
    
    return score
        """,
        """
def final_score(nums):
    # Initialize score
    score = 0
    
    # Continue the process until the matrix becomes empty
    while nums:
        # Find the maximum value from each row
        max_values = [max(row) for row in nums]
        
        # Find the maximum value among all maximum values
        max_val = max(max_values)
        
        # Add the maximum value to the score
        score += max_val
        
        # Remove the maximum value from each row
        for row in nums:
            if max_val in row:
                row.remove(max_val)
        
        # Remove empty rows
        nums = [row for row in nums if row]
    
    return score
        """,
        """
def final_score(nums):
    # Initialize the total score
    total_score = 0
    
    # Continue the process until the matrix becomes empty
    while nums:
        # Find the maximum values in each row
        max_values = [max(row) for row in nums]
        
        # Find the maximum value among all maximum values
        max_val = max(max_values)
        
        # Add the maximum value to the total score
        total_score += max_val
        
        # Remove the rows containing the maximum value
        nums = [row for row in nums if max_val not in row]
    
    return total_score
        """
    ]
    
    # Chocolate Rotation Solutions
    chocolate_sketch_solutions = [
        """
def min_cost_to_collect_chocolates(nums):
    n = len(nums)
    
    # Find the index of the minimum cost chocolate
    min_cost_index = nums.index(min(nums))
    
    # Calculate the total cost of collecting all chocolates
    total_cost = sum(nums)
    
    # Calculate the minimum cost to collect all chocolates
    min_cost = 0
    for i in range(n):
        if i < min_cost_index:
            min_cost += nums[i] * (min_cost_index - i)
        else:
            min_cost += nums[i] * (n - (i - min_cost_index))
    
    return min(min_cost, total_cost)
        """,
        """
def min_cost_to_collect_chocolates(nums):
    n = len(nums)
    
    total_cost = sum(nums)  # Calculate the initial total cost
    
    min_cost = min(nums)  # Find the minimum cost chocolate to start with
    
    min_index = nums.index(min_cost)  # Find the index of the minimum cost chocolate
    
    min_cycles = min((min_index + 1) % n, (n - min_index - 1))  # Calculate the minimum cycles needed to reach the minimum cost chocolate
    
    # Calculate the minimum cost to collect all chocolates
    min_cost_total = min_cost * n + min_cycles * (total_cost - min_cost)
    
    return min_cost_total
        """,
        """
def min_cost_to_collect_chocolates(nums):
    n = len(nums)
    
    # Sort the costs in ascending order
    sorted_nums = sorted(nums)
    
    total_cost = sum(nums)  # Calculate the initial total cost of collecting all chocolates
    min_cost = float('inf')  # Initialize the minimum cost to a large value
    
    for i in range(n):
        current_cost = 0
        for j in range(n):
            # Calculate the cost to change chocolate type j to type ((j + i) mod n)
            current_cost += min(nums[j], nums[(j + i) % n])
        
        min_cost = min(min_cost, current_cost)  # Update the minimum cost
        
        # Check if the current total cost is higher than the initial cost after an operation
        if current_cost > total_cost:
            break  # No need to continue as it will cycle back to the initial state
    
    return min_cost
        """
    ]
    
    chocolate_baseline_solutions = [
        """
def min_cost_to_collect_chocolates(nums):
    n = len(nums)
    
    # Initialize total cost to 0
    total_cost = 0
    
    # Iterate through each type of chocolate
    for i in range(n):
        # Calculate the cost to change all chocolates of type i to type (i+1) mod n
        cost_to_change = min(nums[j] for j in range(n) if j % n == (i + 1) % n)
        
        # Update total cost
        total_cost += cost_to_change
    
    return total_cost
        """,
        """
def min_cost_to_collect_chocolates(nums):
    n = len(nums)
    
    total_cost = sum(nums)  # Total cost to collect all chocolates initially
    
    min_cost = float('inf')  # Initialize minimum cost
    
    for i in range(n):
        curr_cost = 0
        for j in range(n):
            # Calculate the cost to change chocolate of j^th type to ((j + i) mod n)^th type
            curr_cost += nums[j] * min(abs((j + i) % n - j), n - abs((j + i) % n - j))
        
        min_cost = min(min_cost, curr_cost)  # Update minimum cost
    
    return total_cost + min_cost
        """,
        """
def min_cost_to_collect_chocolates(nums):
    n = len(nums)
    
    # Initialize a variable to keep track of the total cost
    total_cost = 0
    
    # Iterate through each index i in the array
    for i in range(n):
        # Calculate the cost to change the chocolate at index i to the next type
        cost_to_next_type = (i + 1) % n
        
        # Calculate the cost to change all chocolates to the next type
        total_cost += min(cost_to_next_type, n - cost_to_next_type) * nums[i]
    
    return total_cost
        """
    ]
    
    # Binary Inversion Solutions
    binary_sketch_solutions = [
        """
def min_cost_to_make_equal(s: str) -> int:
    n = len(s)
    
    # Count the number of '0's and '1's in the string
    zeros = s.count('0')
    ones = n - zeros
    
    # Determine the target state based on counts of '0's and '1's
    target = '0' if zeros > ones else '1'
    
    # Initialize total cost
    total_cost = 0
    
    # Sliding window approach to minimize cost
    cost_left = cost_right = 0
    for i in range(n):
        if s[i] != target:
            # Calculate cost for inverting from left
            cost_left += i + 1
            # Calculate cost for inverting from right
            cost_right += n - i
        
        # Update total cost with the minimum cost for the current position
        total_cost = min(cost_left, cost_right)
    
    return total_cost
        """,
        """
def minCost(s: str) -> int:
    n = len(s)
    zeros = s.count('0')  # Count the number of '0's in the string
    ones = n - zeros  # Calculate the number of '1's in the string

    # Determine the target character based on counts of '0's and '1's
    target_char = '0' if zeros > ones else '1'

    cost = 0
    current_cost = 0
    total_cost = 0

    # Iterate through the string and simulate the sliding window approach
    for i in range(n):
        if s[i] != target_char:
            # Calculate cost for inverting characters starting from the left
            current_cost += 1
        cost += i + 1  # Calculate cost for inverting characters starting from the right
        total_cost += min(cost, current_cost)  # Choose the minimum cost between the two operations

    return total_cost
        """,
        """
def min_cost_to_make_equal(s: str) -> int:
    n = len(s)
    num_zeros = s.count('0')
    num_ones = n - num_zeros

    # Determine the target character based on counts
    target_char = '0' if num_zeros >= num_ones else '1'

    # Initialize cost to 0
    cost = 0

    # Initialize left and right pointers for sliding window approach
    left = 0
    right = n - 1

    while left < right:
        # Move left pointer towards right if character matches target
        while left < n and s[left] == target_char:
            left += 1

        # Move right pointer towards left if character matches target
        while right >= 0 and s[right] == target_char:
            right -= 1

        # Perform the operation with lower cost
        if left < right:
            if left + 1 <= n - right:
                cost += left + 1
                left += 1
            else:
                cost += n - right
                right -= 1

    return cost
        """
    ]
    
    binary_baseline_solutions = [
        """
def minCost(s: str) -> int:
    n = len(s)
    
    # Function to calculate the cost of making all characters in s equal to target_char
    def calculate_cost(target_char: str) -> int:
        cost = 0
        flips = 0
        
        for i in range(n):
            if s[i] != target_char:
                flips += 1
                cost += i + 1
        
        min_cost = cost
        for i in range(n):
            flips += 1
            cost = cost - flips + (n - flips)
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    # Calculate the minimum cost for making all characters equal to '0' and '1'
    cost_0 = calculate_cost('0')
    cost_1 = calculate_cost('1')
    
    return min(cost_0, cost_1)
        """,
        """
def minCost(s: str) -> int:
    n = len(s)
    cost = 0
    flip_count = 0
    
    for i in range(n):
        if s[i] != s[0]:
            flip_count += 1

    # If all characters are already the same, return 0
    if flip_count == 0:
        return 0

    # Initialize the minimum cost with a large value
    min_cost = float('inf')

    # Initialize the count of flipped characters
    flipped = 0

    # Iterate through the string
    for i in range(n):
        if s[i] != s[0]:
            flipped += 1

        # Calculate the cost of flipping all characters before index i
        cost = i + 1

        # Calculate the cost of flipping all characters after index i
        cost += n - i

        # Update the minimum cost
        min_cost = min(min_cost, cost)

        # Update the minimum cost considering flipping all characters before index i
        min_cost = min(min_cost, flipped)

    return min_cost
        """,
        """
def min_cost_to_make_string_equal(s: str) -> int:
    n = len(s)
    cost = 0
    count_ones = s.count('1')
    count_zeros = n - count_ones
    
    # Calculate the cost to make all characters equal to '0'
    cost_to_all_zeros = 0
    curr_zeros = 0
    for i in range(n):
        if s[i] == '1':
            cost_to_all_zeros += curr_zeros
        else:
            curr_zeros += 1
    
    # Calculate the cost to make all characters equal to '1'
    cost_to_all_ones = 0
    curr_ones = 0
    for i in range(n-1, -1, -1):
        if s[i] == '0':
            cost_to_all_ones += curr_ones
        else:
            curr_ones += 1
    
    # Choose the minimum cost between two options
    return min(cost_to_all_zeros, cost_to_all_ones)
        """
    ]
    
    results['row_max_score'] = analyze_similarity('row_max_score', row_max_sketch_solutions, row_max_baseline_solutions)
    results['chocolate_rotation'] = analyze_similarity('chocolate_rotation', chocolate_sketch_solutions, chocolate_baseline_solutions)
    results['binary_inversion'] = analyze_similarity('binary_inversion', binary_sketch_solutions, binary_baseline_solutions)
    
    print("\n" + "=" * 60)
    print("Overall Summary")
    print("=" * 60)
    
    avg_sketch = sum(r['sketch_similarity'] for r in results.values()) / len(results)
    avg_baseline = sum(r['baseline_similarity'] for r in results.values()) / len(results)
    
    print(f"Average sketch-based similarity across all problems: {avg_sketch:.4f}")
    print(f"Average baseline similarity across all problems: {avg_baseline:.4f}")
    print(f"Average difference (sketch - baseline): {avg_sketch - avg_baseline:.4f}")

if __name__ == "__main__":
    main()