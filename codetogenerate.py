#hexa_space = [1st coord, 2nd coord, step, energy, gotEvent]


obstacles = [(3,0), (1,2), (1,3), (0,4), (2,4), (0,6), (1,6), (0,7), (-3,8)]

traps = [(0,1), (3,2), (-1,3), (0,5), (-2,6), (-2,8)]

rewards = [(2,1), (-2,4), (2,5), (-2,7)]

golds = [(2,3), (-1,4), (-2,9), (-1,7)]


events = set(obstacles + traps + rewards)


state_space = []

x=0

print("[\n")

def checkEvent(x,y,x2,y2):
    return (x,y) in events or (x2,y2) in events



for y in range(10):
    even = not y%2
    notLastCol = not y==9

    for count in range(6):
        notFirst = count
        notLast = not (count == 5)

        if (even or notFirst) and notLastCol:
            space = f'[({x}, {y}), ({x-1}, {y+1}), 1, 1, {checkEvent(x,y,x-1,y+1)}],'
            print(space)
        if (not even or notLast) and notLastCol:
            space = f'[({x}, {y}), ({x}, {y+1}), 1, 1, {checkEvent(x,y,x,y+1)}],'
            print(space)
        if notLast:
            space = f'[({x}, {y}), ({x+1}, {y}), 1, 1, {checkEvent(x,y,x+1,y)}],'
            print(space)
        #print()
        x+=1
    if even:
        x-=7
    else:
        x-=6
