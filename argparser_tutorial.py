import argparse
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument("echo", help = "Echo the String here")
parser.add_argument("square", help = "Square of the Given Number", type = int)
parser.add_argument("expo", help = "Get the exponential value", type = int)
parser.add_argument("-v", "--verbosity", help = "Increase output verbosity",
                    action = "store_true")

args = parser.parse_args()

print("String Provided".format(args.echo))
print("Square of {} is {}".format(args.square, args.square**2))
print("Exponential of {} is {}".format(args.expo, np.exp(args.expo)))
if args.verbosity:
    print("Verbosity turned on!!")
