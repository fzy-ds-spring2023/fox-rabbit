"""
    Add argument parsing to fox-rabbit.py
    NOTES: modify this code to do #3 for fox-rabbit.py
    + modify the fox-rabbit.py code to use argparse
        for the params in ALLCAPS at the top of the file
        (e.g., RABBITS, FOXES, etc.)
"""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of person you are greeting", type=str)
    parser.add_argument("num", help="Number of exclamation points", type=int)
    parser.add_argument("-g", "--greeting", help="The greeting")
    parser.add_argument("-s", "--shout", help="Whether to SHOUT the greeting", action="store_true")
    args = parser.parse_args()


    greeting = "Hello"
    if args.greeting:
        greeting = args.greeting

    msg = f"{greeting} {args.name}{'!'*args.num}"
    if args.shout:
        msg = msg.upper()

    print(msg)


main()