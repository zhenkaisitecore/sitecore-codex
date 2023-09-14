import time

# param 1: function name
# param 2 or more, just put in the param
# support positional arguments and keyword arguments
def timeit (func, *args, **kargs):
    start_time = time.time()
    result = func(*args, **kargs)
    return result, (time.time() - start_time)

#test
def Subtract (aNum = 0, bNum = 0):
    time.sleep(0.5)

    return (aNum-bNum)

#test
def main():
    time.sleep(2)
    ans = Subtract(9,3)
    print(ans)
    return ans

    



# print(timeit(Subtract,bNum = 9))
# print(timeit(Subtract, 9, 5))