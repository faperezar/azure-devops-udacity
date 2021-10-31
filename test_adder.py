#En caso que quisiera traerme un function de otro archivo from adder import add
def add(x,y):
    return x+y

def test_add():
    total = add(1,2)
    assert total == 3


#Ejecion python -m pytest -vv test_adder.py