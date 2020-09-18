


param = {'b':2, 'c':3}
print(*param)

def main(a=1,**param):
    print(a)

if __name__ == '__main__':
    main()