def count_images(n):
    # number of possible edges between n dots
    edges = n * (n - 1) // 2
    
    # each edge can be present or not
    total_images = 2 ** edges
    
    return total_images


# test
for n in range(1, 10):
    print(f"N={n}, images={count_images(n)}")