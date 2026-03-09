db = {}

class Client:
    def __init__(self, name):
        self.name = name
        self.booked = []

    def __str__(self):
        return self.name
    

class Seat:
    def __init__(self, seat, client):
        self.seat = seat
        self.client = client


class Airplane:
    def __init__(self): 
        self.seats = {f'{i}{j}': Seat(f'{i}{j}', None) for i in 'ABCDEFGHIJ' for j in range(1, 21)}



