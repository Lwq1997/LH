class Parent:
    pass

class Child(Parent):
    pass

obj = Child()
print(hasattr(obj, 'method'))  # 输出: True