from DeepVision.classification.DenseNet import train

if __name__ == "__main__":
    args = train.arg_parse()
    train.main(args)