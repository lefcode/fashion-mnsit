# Dynamic Programming - Recursive Functions tutorial
# https://www.youtube.com/watch?v=oBt53YbR9Kk&list=WL&index=1&t=2324s

def fibo(n, memo=dict()):
    # computes the fibonacci for a given number
    # time: O(n), space: O(n)
    if n in memo: return memo[n]
    if n <= 2: return 1

    memo[n] = fibo(n-1, memo) + fibo(n-2, memo)
    return memo[n]


def fiboTab(n): # tabulation
    # time: O(n), space: O(n)
    table = [0 for _ in range(n+1)]
    table[1] = 1

    for i in range(n):
        table[i + 1] += table[i]
        try:
            table[i + 2] += table[i]
        except IndexError:
            continue

    return table[n]

def gridTraveler(m, n, memo=dict()):
    # given the dimensions of a grid we find in how many steps can we navigate from top left to bot right
    key = str(m) + "," + str(n)

    if key in memo: return memo[key]

    if m == 0 or n == 0: return 0
    if m == 1 or n == 1: return 1

    memo[key] = gridTraveler(m-1, n, memo) + gridTraveler(m, n-1, memo)
    return memo[key]


def gridTravelerTab(m, n): # tabulation
    # time: O(m*n), space: O(m*n)
    table = [[0 for _ in range(n+1)] for _ in range(m+1)]
    table[1][1] = 1

    for i in range(m+1):
        for j in range(n+1):
            if j+1<=n: table[i][j + 1] += table[i][j]
            if i+1<=m: table[i + 1][j] += table[i][j]

    return table[m][n]


def canSum(targetSum, numbers, memo=dict()): # decision problem
    # given a list of numbers and a targetSum we try to see whether we can create such a combination from the
    # list that produces the targetSum (same numbers allowed) (True or False)
    if targetSum in memo: return memo[targetSum]

    if targetSum == 0: return True
    if targetSum < 0: return False
    if not numbers: return False

    for num in numbers:
        rem = targetSum-num
        memo[rem] = canSum(rem, numbers, memo)
        if memo[rem]:
            return True

    return False


def canSumTab(targetSum, numbers): # tabulation
    # time: O(m*n), space: O(m)
    table = [False for _ in range(targetSum+1)]
    table[0] = True

    for i in range(targetSum+1):
        if table[i]:
            for num in numbers:
                try:
                    table[i + num] = True
                except IndexError:
                    continue

    return table[targetSum]


def howSum(targetSum, numbers, memo=dict()): # combinatoric problem
    # given a list of numbers and a targetSum we return an array that contains any combinations of numbers that
    # add up to the targetSum (return any combination)
    if targetSum in memo: return memo[targetSum]

    if targetSum < 0: return None
    if targetSum == 0: return []
    # if not numbers: return None

    for num in numbers:
        rem = targetSum - num
        memo[targetSum] = howSum(rem, numbers, memo)
        if memo[targetSum] != None:
            memo[targetSum].append(num)
            return memo[targetSum]
    return None


def howSumTab(targetSum, numbers): # tabulation
    # time: O(m^2*n), space: O(m^2)

    table = [None for _ in range(targetSum + 1)]
    table[0] = []

    for i in range(targetSum+1):
        if isinstance(table[i], list):
            for num in numbers:
                try:
                    table[i + num] = table[i].copy()
                    table[i + num].append(num)
                except IndexError:
                    continue

    return table[targetSum]


def bestSum(targetSum, numbers, memo=dict()): # optimization problem
    # given a list of numbers and a targetSum we return the shortest array that contains any combinations of
    # numbers that add up to the targetSum (return any combination)
    # if targetSum in memo: return memo[targetSum]

    if targetSum < 0: return None
    if targetSum == 0: return []
    # if not numbers: return None
    shortest_comb = None

    for num in numbers:
        rem = targetSum - num
        rem_result = bestSum(rem, numbers, memo)

        if rem_result != None:
            rem_result.append(num)
            comb = rem_result
            if shortest_comb == None or len(comb) < len(shortest_comb):
                shortest_comb = comb

    # memo[targetSum] = shortest_comb
    return shortest_comb


def bestSumTab(targetSum, numbers):# tabulation
    # time: O(m^2*n), space: O(m^2)
    table = [None for _ in range(targetSum + 1)]
    table[0] = []

    for i in range(targetSum+1):
        if isinstance(table[i], list):
            for num in numbers:
                try:
                    new_val = table[i].copy()
                    new_val.append(num)

                    if table[i + num]:
                        if len(new_val) < len(table[i+num]):
                            table[i + num] = new_val
                    else:
                        table[i + num] = new_val

                except IndexError:
                    continue

    return table[targetSum]

print(bestSumTab(100, [1, 2, 5, 25]))


def canConstruct(target, wordBank, memo=dict()):
    # return if target can be constructed by concatenating workBank elements (reuse allowed) (True or False)
    if target in memo: return memo[target]

    if target == "": return True
    if wordBank == []: return False

    for word in wordBank:
        if target.find(word) == 0:
            suffix = target[len(word):]
            memo[target] = countConstruct(suffix, wordBank)
            if memo[target]:
                return True

    memo[target] = False
    return False


def countConstruct(target, wordBank, memo=dict()):
    # return the number of ways the target can be constructed by wordBank
    if target in memo: return memo[target]

    if target == "": return 1
    if wordBank == []: return 0

    ways_count = 0
    for word in wordBank:
        if target.find(word) == 0:
            suffix = target[len(word):]
            ways_count += countConstruct(suffix, wordBank, memo)

    memo[target] = ways_count
    return ways_count


def allConstruct(target, wordBank):
    # return an array containing all possible ways to construct target from the wordBank

    if target == "": return [[]]
    #if target != "": return []
    # if wordBank == []: return None

    result = []

    for word in wordBank:
        if target.find(word) == 0:
            suffix = target[len(word):]
            suffix_ways = allConstruct(suffix, wordBank)
            print(suffix_ways)
            target_ways = list(map((lambda x: x.append(word)), suffix_ways))
            result.append(target_ways)
    return result


if __name__ == "__main__":
    pass
    #allConstruct("trash", ["ash", "tr", "as"])
    #print(allConstruct("purple", ["purp", "p", "ur", "le", "purpl"]))