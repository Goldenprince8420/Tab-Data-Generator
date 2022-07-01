import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("frame_path", help = "Reference Dataframe", type = str)
parser.add_argument("S", help = "Protected Attribute", type = str)
parser.add_argument("Y", help = "Labels", type = str)
parser.add_argument("up_value", help = "Under-Privileged Group Value", type = str)
parser.add_argument("d_value", help = "Desirable Value", type = str)

# Run Parameters
parser.add_argument("num_epochs", help = "Number of Epochs", type = int)
parser.add_argument("batch_size", help = "Number of Batches to divide into", type = int)
parser.add_argument("num_fair_epochs", help = "Number of fair Training Epochs", type = int)
parser.add_argument("lambda_param", help = "Value of Lambda Parameter", type = float)
parser.add_argument("gen_file", help = "Name of Generated File", type = str)
parser.add_argument("gen_data_size", help = "Number of Examples to Generate", type = str)

args = parser.parse_args()


def process_arguments():
    PARAMS['S'] = args.S
    PARAMS['Y'] = args.Y
    PARAMS['S_under'] = args.up_value
    PARAMS['Y_desire'] = args.d_value
    PARAMS['file_name'] = args.frame_path
    data = pd.read_csv(PARAMS['file_name'])
    data[S] = data[S].astype(object)
    data[Y] = data[Y].astype(object)
    PARAMS['data'] = data
    return PARAMS


def main():
    params = process_arguments()
    return params


if __name__ == "__main__":
    PARAMS = main()
    print("Done!!")
