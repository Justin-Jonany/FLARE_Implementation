class a:
    def __init__(self) -> None:
        self.xy = 3
    def special_print(self):
        print(self.xy)
        print('% s')

a_obj = a()

a_obj.special_print()