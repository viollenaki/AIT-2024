
class Magazine:
    def __init__(self):
        self.base = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        self.m = len(self.base)

    def next_or_equal_power_of_two(self,n):
        import math
        if n <= 1:
            return 0

        power = math.log2(n)

        if power.is_integer():
            return int(power)
        
        return math.ceil(power)

    def get_symbols_code_sum_in_binary(self,symbols):
        return bin(sum((ord(symbol) for symbol in symbols)))

    def get_index(self, prod):
        return int(f'{self.get_symbols_code_sum_in_binary(prod)}'[-(self.next_or_equal_power_of_two(self.m))-1:],2)
    
    def write_into_base(self,prod):
        index = self.get_index(prod)
        if len(self.base[index]) == 0:
            self.base[index] = prod
            return 'written'
        else:
            for i in range(index+1, len(self.base)):
                if len(self.base[i]) == 0:
                    self.base[i] = prod
                    return 'written'
            for i in range(index):
                if len(self.base[i]) == 0:
                    self.base[i] = prod
                    return 'written'
        return 'there is not free space in base'

magazine = Magazine()
products = [
    "Product1", "Product2", "Product3", "Product4", "Product5", 
    "Product6", "Product7", "Product8", "Product9", "Product10", 
    "Product11", "Product12", "Product13", "Product14", "Product15", 
    "Product16", "Product17", "Product18", "Product19", "Product20", 
    "Product21", "Product22", "Product23"
]


for product in products:
    result = magazine.write_into_base(product)
    print(f'Adding {product}: {result}')








